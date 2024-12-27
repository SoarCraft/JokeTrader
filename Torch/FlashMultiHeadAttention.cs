namespace JokeTrader.Torch;

using TorchSharp;
using TorchSharp.FlashAttention;
using TorchSharp.Modules;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

internal class FlashMultiHeadAttention : Module<Tensor, Tensor> {
    private readonly int numHeads;

    private readonly int headDim;

    private readonly Linear qkvProj;

    private readonly Linear output;

    private readonly FlashAttention flashAttention;

    public FlashMultiHeadAttention(int embedDim, int numHeads, float dropoutRate = 0.1f) : base(nameof(FlashMultiHeadAttention)) {
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        this.qkvProj = Linear(embedDim, embedDim * 3);
        this.output = Linear(embedDim, embedDim);

        this.flashAttention = new(
            1f / MathF.Sqrt(this.headDim),
            dropoutRate,
            false
        );

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input) {
        using var _ = NewDisposeScope();

        var batchSize = input.shape[0];
        var seqLen = input.shape[1];

        var qkv = this.qkvProj.forward(input);
        qkv = qkv.view(batchSize, seqLen, 3, this.numHeads, this.headDim);

        var context = this.flashAttention.forward(qkv);
        context = context.view(batchSize, seqLen, -1);

        return this.output.forward(context).MoveToOuterDisposeScope();
    }
}
