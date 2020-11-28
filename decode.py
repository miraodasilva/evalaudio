import torch
import torch.nn as nn
import ctcdecode


class CtcDecoder:
    """
        Decode sentence from ctc networks outputs
    """

    def __init__(self, alphabet, beam_width=20):
        self.alphabet = alphabet
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(alphabet, model_path=None, beam_width=beam_width,
                                                    blank_id=alphabet.index('_'), alpha=1, beta=1.5)
        self.softmax = nn.Softmax(dim=2)
        self.beam_width = beam_width

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])

    def decode(self, probs):
        result, score, timestep, out_seq_len = self.ctc_decoder.decode(probs)
        return self.convert_to_string(result[0][0], self.alphabet, out_seq_len[0][0])

    def decode_batch(self, batch_probs, length=None):
        batch_probs = self.softmax(batch_probs).cpu().detach()

        decoded = []
        for j in range(batch_probs.shape[1]):
            nout = torch.Tensor(batch_probs[:length[j], j, :])  # length[j] length[j]
            nout = nout.view(1, nout.shape[0], batch_probs.shape[2]).float()
            d = self.decode(nout)
            decoded.append(d)
        return decoded
