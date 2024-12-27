namespace JokeTrader.Torch;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class PositionalEncoding() : Module<Tensor, Tensor>(nameof(PositionalEncoding)) {
    public override Tensor forward(Tensor input) {
        using var _ = NewDisposeScope();

        var batchSize = input.shape[0];
        var seqLen = input.shape[1];
        var embedDim = input.shape[2];

        var intervals = input[.., .., -1];
        var features = input[.., .., ..^1];

        var posEncoding = zeros(batchSize, seqLen, embedDim, device: input.device);
        for (var i = 0; i < (embedDim - 1) / 2; i++) {
            var div = exp(-log(10000.0) * (2 * i) / embedDim);
            posEncoding[.., .., 2 * i] = sin(intervals.unsqueeze(-1) * div);
            posEncoding[.., .., 2 * i + 1] = cos(intervals.unsqueeze(-1) * div);
        }

        return (features + posEncoding).MoveToOuterDisposeScope();
    }
}
