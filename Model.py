# Code framework for "Zoneformer: On-device Neural Beamformer For In-car Multi-zone Speech Separation, Enhancement and Echo Cancellation" 
# Anonymous submission to INTERSPEECH 2023
from conf import *
import torch as th
import torch.nn as nn
import functional as FC
from beamformer_cm import *
from tensor import ComplexTensor
import torch.nn.functional as Fn
from audio_fea import DFComputer, iSTFT
import math
from typing import List, Tuple, Optional
import numpy as np
import librosa

# class for splitting linear spectrum into Mel-subband according to the traditional Mel-scale
class MelGroupConv(nn.Module):
    def __init__(self, in_channels=257, n_subbands=16, subband_dim=8, samplerate=16000):
        super(MelGroupConv, self).__init__()

        freq_split  = librosa.mel_frequencies(n_mels=n_subbands+1,fmin=0.0, fmax=samplerate/2)
        freq_split  = np.array([int(np.round(i*2*in_channels/samplerate)) for i in freq_split])
        self.groups = freq_split[1:] - freq_split[:-1]
        ratio = np.round(100*np.mean(self.groups<=subband_dim),2)
        hidden_dim  = int(2**np.ceil(np.log2(self.groups[-1])))
        self.conv_modules = []
        for i in range(len(self.groups)):
            mel_conv = nn.Sequential(nn.Conv2d(self.groups[i], hidden_dim, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
                                     nn.Conv2d(hidden_dim, subband_dim, kernel_size=(3,1), stride=(1,1), padding=(1,0)))
            self.conv_modules.append(mel_conv)
        self.conv_modules = nn.ModuleList(self.conv_modules)

    def split(self, x):
        xsplit = []
        start = 0
        end   = 0
        for i in range(len(self.groups)):
            end = end + self.groups[i]
            xsplit.append(x[:,start:end,...])
            start = end
        return xsplit

    def forward(self, x):
        xsplit = self.split(x)
        output = th.stack([self.conv_modules[i](xsplit[i]) for i in range(len(xsplit))],1)
        return output

# class for inverse Mel to linear spectrum
class InvMelGroupConv(nn.Module):
    def __init__(self, n_subbands=16, subband_dim=8, out_channels=257, samplerate=16000):
        super(InvMelGroupConv, self).__init__()

        freq_split  = librosa.mel_frequencies(n_mels=n_subbands+1,fmin=0.0, fmax=samplerate/2)
        freq_split  = np.array([int(np.round(i*2*out_channels/samplerate)) for i in freq_split])
        self.groups = freq_split[1:] - freq_split[:-1]
        ratio = np.round(100*np.mean(self.groups<=subband_dim),2)
        hidden_dim  = int(2**np.ceil(np.log2(self.groups[-1])))
        self.conv_modules = []
        for i in range(len(self.groups)):
            mel_conv = nn.Sequential(nn.Conv2d(subband_dim, hidden_dim, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
                                     nn.Conv2d(hidden_dim, self.groups[i], kernel_size=(3,1), stride=(1,1), padding=(1,0)))
            self.conv_modules.append(mel_conv)
        self.conv_modules = nn.ModuleList(self.conv_modules)

    def forward(self, x):
        n_subbands = x.shape[1]
        output = th.cat([self.conv_modules[i](x[:,i]) for i in range(n_subbands)],1)
        return output

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

# Adapted from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """
    
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)
    
    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x

class RNNBF(nn.Module):
    """
    Proposed Mel-subband multi-head self-attentive RNN Beamformer
    """

    def __init__(
            self,
            # audio conf
            L=512,
            N=256,
            X=8,
            R=4,
            B=256,
            H=512,
            P=3,
            F=256,
            cos=True,
            ipd=""):
        super(RNNBF, self).__init__()

        self.cos = cos

        self.df_computer = DFComputer(frame_hop=L // 2,
                                      frame_len=L,
                                      in_feature=['LPS', 'IPD', 'AF'],
                                      merge_mode='sum',
                                      cosIPD=True,
                                      sinIPD=True,
                                      speaker_feature_dim=4)
        
        ################## define filtering size #######################
        self.filter_size = 2 # 2*1 cRF filter, i.e., +-1 time/freq
        rnn_hdim=128    
        n_mic=2+1 #2-channel + echo-channel
        self.hid_dim=128
        n_tap=1
        self.n_band = 64 # num of Mel-band
        self.band_width = math.ceil(self.df_computer.num_bins/self.n_band) # self.df_computer.num_bins
        
        # Full-band audio encoder (Audio_Enc)
        self.Dense_linear = nn.Linear(self.df_computer.num_bins*11, rnn_hdim) # 11 features = 2-channel lps, echo lps, 4 d(\theta), 2 cosIPD, 2 sinIPD
        self.Audio_Enc=nn.GRU(rnn_hdim,rnn_hdim,1,batch_first=True)
      
        # get speech cRF
        self.crf_speech_real = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)
        self.crf_speech_imag = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)
        
        # get noise cRF
        self.crf_noise_real = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)
        self.crf_noise_imag = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)
   
        # get echo cRF
        self.crf_echo_real = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)
        self.crf_echo_imag = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)

        #get echo_v2 cRF 
        self.crf_echo_v2_real = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)
        self.crf_echo_v2_imag = Conv1D(rnn_hdim, self.df_computer.num_bins*(self.filter_size), 1)

        self.subband_split = MelGroupConv(in_channels=self.df_computer.num_bins, n_subbands=self.n_band, subband_dim=self.band_width)
        
        self.psd_matrix_ln=nn.LayerNorm([n_mic*n_mic*n_tap*2*2*self.band_width],elementwise_affine=True)
        self.hid_ln=nn.LayerNorm([self.hid_dim],elementwise_affine=True)
        
        self.Dense_1 = nn.Linear(n_mic*n_mic*n_tap*2*2*self.band_width + self.hid_dim, self.hid_dim)
        self.GRU_h1= nn.GRU(self.hid_dim,self.hid_dim,1,batch_first=True)
        self.Dense_2=nn.Linear(self.hid_dim*2,self.hid_dim)
        self.GRU_h2= nn.GRU(self.hid_dim,self.hid_dim,1,batch_first=True) 
        ## self-attention layer to learn high order correlation among channels
        self.multi_head_attention=Attention(embed_dim=self.hid_dim, num_heads=4)

        self.Dense_out = nn.Linear(self.hid_dim,2*2*4*self.band_width)
        self.subband_stitch = InvMelGroupConv(out_channels=self.df_computer.num_bins, n_subbands=self.n_band, subband_dim=self.band_width)
        self.istft = iSTFT(frame_len=L, frame_hop=L // 2, num_fft=L)
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x, echo, directions, spk_num):
        """
        x: raw waveform chunks, N x C
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 2/3D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        
        # when inference, only one utt
        if x.dim() == 2:
            x = th.unsqueeze(x, 0)
            echo = th.unsqueeze(echo, 0)
            spk_num = th.unsqueeze(spk_num, 0)
            directions = th.unsqueeze(directions, 0)
       
        audio_fea, mag, phase, real, imag, echo_real, echo_imag, stv_real, stv_imag = self.df_computer([x, echo, directions, spk_num]) # steering vector [Batch, n_channel, nspk, n_bins]
        echo_real=th.unsqueeze(echo_real,1) 
        echo_imag=th.unsqueeze(echo_imag, 1)
        B_size = audio_fea.size(0)
        F_size = audio_fea.size(1)
        T_size = audio_fea.size(2)
        D_size = audio_fea.size(3)        
        a = th.transpose(audio_fea, 1, 2)
        a = a.contiguous()
        a = a.view(B_size, T_size, F_size*D_size)
        #a = self.quant(a)
        a = self.Dense_linear(a) #linear layer dim reduction
        #a = self.dequant(a)
        a,_ = self.Audio_Enc(a)
        G_spec  = a     

        ################# filtering #################
        # target speech cRM filters
        a = th.transpose(a, 1, 2)
        a=a.contiguous()
        m_targ = self.crf_speech_real(a)
        m_targ_imag = self.crf_speech_imag(a)
        m_targ = th.transpose(m_targ,1,2)
        m_targ_imag = th.transpose(m_targ_imag,1,2)
        m_targ = m_targ.view(m_targ.size(0),m_targ.size(1),self.df_computer.num_bins,-1)
        m_targ = th.transpose(m_targ,1,2)
        m_targ_imag = m_targ_imag.view(m_targ_imag.size(0),m_targ_imag.size(1),self.df_computer.num_bins,-1)
        m_targ_imag = th.transpose(m_targ_imag,1,2)
        m_targ_imag += 1e-10
        m_targ_complex = ComplexTensor(m_targ,m_targ_imag)

        # noise cRM filters
        m_noise = self.crf_noise_real(a)
        m_noise_imag = self.crf_noise_imag(a) 
        m_noise = th.transpose(m_noise,2,1)
        m_noise = m_noise.view(m_noise.size(0),m_noise.size(1),self.df_computer.num_bins,-1)
        m_noise = th.transpose(m_noise,1,2)
        m_noise_imag = th.transpose(m_noise_imag,2,1)
        m_noise_imag = m_noise_imag.view(m_noise_imag.size(0),m_noise_imag.size(1),self.df_computer.num_bins,-1)
        m_noise_imag = th.transpose(m_noise_imag,1,2)
        m_noise_imag += 1e-10
        m_noise_complex = ComplexTensor(m_noise,m_noise_imag)


        # echo cRM filters
        m_echo = self.crf_echo_real(a)
        m_echo_imag = self.crf_echo_imag(a)
        m_echo = th.transpose(m_echo,2,1)
        m_echo = m_echo.view(m_echo.size(0),m_echo.size(1),self.df_computer.num_bins,-1)
        m_echo = th.transpose(m_echo,1,2)
        m_echo_imag = th.transpose(m_echo_imag,2,1)
        m_echo_imag = m_echo_imag.view(m_echo_imag.size(0),m_echo_imag.size(1),self.df_computer.num_bins,-1)
        m_echo_imag = th.transpose(m_echo_imag,1,2)
        m_echo_imag += 1e-10
        m_echo_complex = ComplexTensor(m_echo, m_echo_imag)

        # echo_v2 cRF filters
        m_echo_v2 = self.crf_echo_v2_real(a)
        m_echo_v2_imag = self.crf_echo_v2_imag(a)
        m_echo_v2 = th.transpose(m_echo_v2,2,1)
        m_echo_v2 = m_echo_v2.view(m_echo_v2.size(0),m_echo_v2.size(1),self.df_computer.num_bins,-1)
        m_echo_v2 = th.transpose(m_echo_v2,1,2)
        m_echo_v2_imag = th.transpose(m_echo_v2_imag,2,1)
        m_echo_v2_imag = m_echo_v2_imag.view(m_echo_v2_imag.size(0),m_echo_v2_imag.size(1),self.df_computer.num_bins,-1)
        m_echo_v2_imag = th.transpose(m_echo_v2_imag,1,2)
        m_echo_v2_imag += 1e-10
        m_echo_v2_complex = ComplexTensor(m_echo_v2, m_echo_v2_imag)       

        # Time and frequency shift on y to make 2X1 filtering
        real_tf_shift = th.stack((th.roll(real,0,dims=3),th.roll(real,1,dims=3)),4)
        imag_tf_shift = th.stack((th.roll(imag,0,dims=3),th.roll(imag,1,dims=3)),4)
        imag_tf_shift += 1e-10
        real_tf_shift = th.transpose(real_tf_shift,3,4)
        imag_tf_shift = th.transpose(imag_tf_shift,3,4) + 1.0e-10
        y_complex = ComplexTensor(real_tf_shift, imag_tf_shift)

        #get tf rolled echo for cRF filtering
        echo_real_tf_shift = th.stack((th.roll(echo_real,0,dims=3),th.roll(echo_real,1,dims=3)),4)
        echo_imag_tf_shift = th.stack((th.roll(echo_imag,0,dims=3),th.roll(echo_imag,1,dims=3)),4)
        echo_imag_tf_shift += 1e-10
        echo_real_tf_shift = th.transpose(echo_real_tf_shift,3,4)
        echo_imag_tf_shift = th.transpose(echo_imag_tf_shift,3,4) + 1.0e-10
        echo_complex = ComplexTensor(echo_real_tf_shift, echo_imag_tf_shift)

        # Apply 3X3 cRM filtering to get est_speech and est_noise
        est_speech_complex = apply_cRM_filter(m_targ_complex, y_complex)
        est_real_part = est_speech_complex.real 
        est_imag_part = est_speech_complex.imag + 1.0e-10

        est_noise_complex = apply_cRM_filter(m_noise_complex, y_complex)
        est_real_noise = est_noise_complex.real 
        est_imag_noise = est_noise_complex.imag + 1.0e-10

        #apply cRF to get est_echo:
        est_echo_complex = apply_cRM_filter(m_echo_complex, echo_complex)
        est_real_echo = est_echo_complex.real
        est_imag_echo = est_echo_complex.imag + 1.0e-10   

        #apply cRF to get est_echo_v2
        est_echo_v2_complex = apply_cRM_filter(m_echo_v2_complex, echo_complex)
        est_real_echo_v2 = est_echo_v2_complex.real
        est_imag_echo_v2 = est_echo_v2_complex.imag + 1.0e-10

        ###concatenate est_echo with est_speech or est_noise
        est_real_part=th.cat([est_real_part, est_real_echo], 1)
        est_imag_part=th.cat([est_imag_part, est_imag_echo], 1)
        est_real_noise=th.cat([est_real_noise, est_real_echo_v2],1)
        est_imag_noise=th.cat([est_imag_noise, est_imag_echo_v2],1)

        #########################beamforming part 
        mc_real=th.transpose(real,1,2) #[B,F,C,T]
        mc_imag=th.transpose(imag,1,2)
        mc_complex = ComplexTensor(mc_real,mc_imag)        
        
        est_real_part = th.transpose(est_real_part,1,2)
        est_imag_part = th.transpose(est_imag_part,1,2)
        est_real_part = est_real_part.contiguous()
        est_imag_part = est_imag_part.contiguous()
        est_speech_complex = ComplexTensor(est_real_part, est_imag_part)

        est_real_noise = th.transpose(est_real_noise,1,2)
        est_imag_noise = th.transpose(est_imag_noise,1,2)
        est_real_noise = est_real_noise.contiguous()
        est_imag_noise = est_imag_noise.contiguous()
        est_noise_complex=ComplexTensor(est_real_noise, est_imag_noise) 
 
        psd_speech = get_power_spectral_density_matrix_self_with_cm_t(est_speech_complex)
        psd_noise = get_power_spectral_density_matrix_self_with_cm_t(est_noise_complex) #[B,F,T,C,C]

        cat_psdni_psds_flatten=th.cat([psd_speech.real.contiguous().view(B_size, F_size, T_size, -1), psd_speech.imag.contiguous().view(B_size, F_size, T_size, -1), psd_noise.real.contiguous().view(B_size, F_size, T_size, -1), psd_noise.imag.contiguous().view(B_size, F_size, T_size, -1)],dim=-1)
        
        cat_psdni_psds_flatten_mel_split =self.subband_split(cat_psdni_psds_flatten)
        cat_psdni_psds_flatten = th.transpose(cat_psdni_psds_flatten_mel_split, 2, 3) 
        cat_psdni_psds_flatten = cat_psdni_psds_flatten.contiguous()
        cat_psdni_psds_flatten = cat_psdni_psds_flatten.view(B_size*self.n_band, T_size, self.band_width*cat_psdni_psds_flatten.size(-1))

        cat_psdni_psds_flatten=self.psd_matrix_ln(cat_psdni_psds_flatten)
        G_spec = th.unsqueeze(G_spec, 1)
        G_spec = G_spec.repeat(1, self.n_band, 1, 1)
        G_spec = G_spec.view(B_size*self.n_band, T_size, G_spec.size(-1))
        cat_psdni_psds_flatten=th.cat([cat_psdni_psds_flatten, G_spec],dim=-1)

        ws_per_frame= Fn.leaky_relu(self.Dense_1(cat_psdni_psds_flatten))
        ws_per_frame = self.hid_ln(ws_per_frame)
        ws_per_frame,_ = self.GRU_h1(ws_per_frame) #

        tmp_ws_per_frame = ws_per_frame.view(B_size, self.n_band, T_size, self.hid_dim)
        G_spatial = th.mean(tmp_ws_per_frame, dim=1, keepdim=True)
        G_spatial = G_spatial.repeat(1,self.n_band,1,1)     
        G_spatial = G_spatial.view(B_size*self.n_band, T_size, self.hid_dim) #[B*self.n_band, T, D]

        ws_per_frame = th.cat([ws_per_frame,G_spatial], dim=-1) #[B*F, T, 2C]
        ws_per_frame = self.Dense_2(ws_per_frame) # downsample 2C to C [B*F, T, C]
        ws_per_frame,_ = self.GRU_h2(ws_per_frame)
 
        #self-attention : multi-head with causal mask
        i, j = ws_per_frame.shape[1],ws_per_frame.shape[1]
        causal_mask = th.ones(i, j, device = ws_per_frame.device).triu_(diagonal=1)
        att_win = 100
        for t in range(i):
            if t > att_win:
                causal_mask[t, 0:(t-att_win)]=1.0
            if t <= att_win: 
                causal_mask[t, 0:att_win]=0.0
        causal_mask =  causal_mask * -10000.0
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = causal_mask.repeat(ws_per_frame.size(0), 1, 1, 1)
        ws_per_frame, attn_weights, _ = self.multi_head_attention(ws_per_frame, attention_mask=causal_mask, output_attentions=False)
        ws_per_frame = Fn.leaky_relu(ws_per_frame)
        ws_per_frame=ws_per_frame.view(B_size, self.n_band, T_size, self.hid_dim)

        ws_per_frame= self.Dense_out(ws_per_frame)
        ws_per_frame = ws_per_frame.view(B_size, self.n_band, T_size, self.band_width, ws_per_frame.size(-1)//self.band_width)  
        ws_per_frame = th.transpose(ws_per_frame,2,3) 
        ws_per_frame = ws_per_frame.contiguous()
        ws_per_frame   = self.subband_stitch(ws_per_frame)
       
        print(ws_per_frame.size())
        mic_size = 2
        #### for left front : co-driver
        ws_per_frame_real_left=ws_per_frame[:,:,:,:mic_size]
        ws_per_frame_imag_left=ws_per_frame[:,:,:,mic_size:2*mic_size]
        #beam pattern constraint using steering vec
        stv_real = th.transpose(stv_real, 1,3)
        stv_imag = th.transpose(stv_imag, 1,3) 
        stv_left_real = th.unsqueeze(stv_real[:,:,1,:],-1) #[B, F, C, 1]
        stv_left_real = stv_left_real.repeat(1,1,1,T_size) #[B, F, C, T]
        stv_left_imag = th.unsqueeze(stv_imag[:,:,1,:],-1)
        stv_left_imag = stv_left_imag.repeat(1,1,1,T_size) # [B,F,C,T]
        stv_left_complex = ComplexTensor(stv_left_real, stv_left_imag) #[B,F,C,T]        
        #############################################
        ws_per_frame_complex_left=ComplexTensor(ws_per_frame_real_left,ws_per_frame_imag_left) 
        print(ws_per_frame_complex_left.size(), stv_left_complex.size())
        beampat_left_loss = apply_beamforming_vector(ws_per_frame_complex_left, stv_left_complex) # -> [B, F, T]
        mean_beampat_left_loss = th.mean(beampat_left_loss.abs(), -2, keepdim=True)
        beamloss_left = (th.abs(beampat_left_loss.abs() - mean_beampat_left_loss))
        bf_enhanced_left = apply_beamforming_vector(ws_per_frame_complex_left, mc_complex) # mc_complex (B,F,C*2,T)
        ##########################################
          
        
        #### for left back: back of the co-driver
        ws_per_frame_real_left_back=ws_per_frame[:,:,:,2*mic_size:3*mic_size]
        ws_per_frame_imag_left_back=ws_per_frame[:,:,:,3*mic_size:4*mic_size]
        ws_per_frame_complex_left_back=ComplexTensor(ws_per_frame_real_left_back,ws_per_frame_imag_left_back)
        #############stv beam loss
        stv_left_back_real = th.unsqueeze(stv_real[:,:,3,:],-1) #[B, F, C, 1]
        stv_left_back_real = stv_left_back_real.repeat(1,1,1,T_size) #[B, F, C, T]
        stv_left_back_imag = th.unsqueeze(stv_imag[:,:,3,:],-1)
        stv_left_back_imag = stv_left_back_imag.repeat(1,1,1,T_size) # [B,F,C,T]
        stv_left_back_complex = ComplexTensor(stv_left_back_real, stv_left_back_imag) #[B,F,C,T]  
        beampat_left_back_loss = apply_beamforming_vector(ws_per_frame_complex_left_back, stv_left_back_complex) # -> [B, F, T]
        mean_beampat_left_back_loss = th.mean(beampat_left_back_loss.abs(),-2,keepdim=True)
        beamloss_left_back = (th.abs(beampat_left_back_loss.abs() - mean_beampat_left_back_loss))
        ##################################################
        bf_enhanced_left_back = apply_beamforming_vector(ws_per_frame_complex_left_back, mc_complex) # mc_complex (B,F,C*2,T)
        ##########################################
        

        #### for right front: main driver
        ws_per_frame_real_right=ws_per_frame[:,:,:,4*mic_size:5*mic_size]
        ws_per_frame_imag_right=ws_per_frame[:,:,:,5*mic_size:6*mic_size]       
        ws_per_frame_complex_right=ComplexTensor(ws_per_frame_real_right,ws_per_frame_imag_right)
        #############stv beam loss
        stv_right_real = th.unsqueeze(stv_real[:,:,0,:],-1) #[B, F, C, 1]
        stv_right_real = stv_right_real.repeat(1,1,1,T_size) #[B, F, C, T]
        stv_right_imag = th.unsqueeze(stv_imag[:,:,0,:],-1)
        stv_right_imag = stv_right_imag.repeat(1,1,1,T_size) # [B,F,C,T]
        stv_right_complex = ComplexTensor(stv_right_real, stv_right_imag) #[B,F,C,T]  
        beampat_right_loss = apply_beamforming_vector(ws_per_frame_complex_right, stv_right_complex) # -> [B, F, T]        
        mean_beampat_right_loss = th.mean(beampat_right_loss.abs(), -2, keepdim=True)
        beamloss_right = (th.abs(beampat_right_loss.abs() - mean_beampat_right_loss))
        ##################################################
        bf_enhanced_right = apply_beamforming_vector(ws_per_frame_complex_right, mc_complex) # mc_complex (B,F,C*2,T)
        ##########################################
        
        #### for right-back : back of main driver
        ws_per_frame_real_right_back=ws_per_frame[:,:,:,6*mic_size:7*mic_size]
        ws_per_frame_imag_right_back=ws_per_frame[:,:,:,7*mic_size:8*mic_size]
        ws_per_frame_complex_right_back=ComplexTensor(ws_per_frame_real_right_back,ws_per_frame_imag_right_back)
        #############stv beam loss
        stv_right_back_real = th.unsqueeze(stv_real[:,:,2,:],-1) #[B, F, C, 1]
        stv_right_back_real = stv_right_back_real.repeat(1,1,1,T_size) #[B, F, C, T]
        stv_right_back_imag = th.unsqueeze(stv_imag[:,:,2,:],-1)
        stv_right_back_imag = stv_right_back_imag.repeat(1,1,1,T_size) # [B,F,C,T]
        stv_right_back_complex = ComplexTensor(stv_right_back_real, stv_right_back_imag) #[B,F,C,T]  
        beampat_right_back_loss = apply_beamforming_vector(ws_per_frame_complex_right_back, stv_right_back_complex) # -> [B, F, T]      
        mean_beampat_right_back_loss = th.mean(beampat_right_back_loss.abs(), -2, keepdim=True)
        beamloss_right_back = (th.abs(beampat_right_back_loss.abs() - mean_beampat_right_back_loss))
        ##################################################
        bf_enhanced_right_back = apply_beamforming_vector(ws_per_frame_complex_right_back, mc_complex) # mc_complex (B,F,C*2,T)
        ##########################################
        
        bf_enhanced_real_left=bf_enhanced_left.real
        bf_enhanced_new_imag_left=bf_enhanced_left.imag +1.0e-10
        bf_enhanced_new_left=ComplexTensor(bf_enhanced_real_left, bf_enhanced_new_imag_left)
        bf_enhanced_mag_left=bf_enhanced_new_left.abs()
        bf_enhanced_angle_left=bf_enhanced_new_left.angle()      
        est_left = self.istft(bf_enhanced_mag_left, bf_enhanced_angle_left, squeeze=False)        
        est_left =th.squeeze(est_left,dim=1)
        
        bf_enhanced_real_left_back=bf_enhanced_left_back.real
        bf_enhanced_new_imag_left_back=bf_enhanced_left_back.imag +1.0e-10
        bf_enhanced_new_left_back=ComplexTensor(bf_enhanced_real_left_back, bf_enhanced_new_imag_left_back)
        bf_enhanced_mag_left_back=bf_enhanced_new_left_back.abs()
        bf_enhanced_angle_left_back=bf_enhanced_new_left_back.angle()
        est_left_back = self.istft(bf_enhanced_mag_left_back, bf_enhanced_angle_left_back, squeeze=False)
        est_left_back =th.squeeze(est_left_back,dim=1)
        

        bf_enhanced_real_right=bf_enhanced_right.real
        bf_enhanced_new_imag_right=bf_enhanced_right.imag +1.0e-10
        bf_enhanced_new_right=ComplexTensor(bf_enhanced_real_right, bf_enhanced_new_imag_right)
        bf_enhanced_mag_right=bf_enhanced_new_right.abs()
        bf_enhanced_angle_right=bf_enhanced_new_right.angle()
        est_right = self.istft(bf_enhanced_mag_right, bf_enhanced_angle_right, squeeze=False)
        est_right =th.squeeze(est_right,dim=1)
        
        bf_enhanced_real_right_back=bf_enhanced_right_back.real
        bf_enhanced_new_imag_right_back=bf_enhanced_right_back.imag +1.0e-10
        bf_enhanced_new_right_back=ComplexTensor(bf_enhanced_real_right_back, bf_enhanced_new_imag_right_back)
        bf_enhanced_mag_right_back=bf_enhanced_new_right_back.abs()
        bf_enhanced_angle_right_back=bf_enhanced_new_right_back.angle()
        est_right_back = self.istft(bf_enhanced_mag_right_back, bf_enhanced_angle_right_back, squeeze=False)
        est_right_back =th.squeeze(est_right_back,dim=1)
        
        return beamloss_left, beamloss_left_back, beamloss_right, beamloss_right_back, est_left, bf_enhanced_mag_left, est_right, bf_enhanced_mag_right,  est_left_back, bf_enhanced_mag_left_back, est_right_back, bf_enhanced_mag_right_back
