namespace JokeTrader.Torch;

using TorchSharp.FlashAttention;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class FlashMultiHeadAttention : Module<Tensor, Tensor> {
    private readonly FlashAttention flashAttention;

    private readonly int headDim;
    private readonly int numHeads;

    private readonly Linear output;

    private readonly Linear qkvProj;

    public FlashMultiHeadAttention(int embedDim, int numHeads, float dropoutRate = 0.1f)
        : base(nameof(FlashMultiHeadAttention)) {
        if (embedDim <= 0)
            throw new ArgumentException("嵌入维度必须大于0", nameof(embedDim));

        if (numHeads <= 0)
            throw new ArgumentException("注意力头数必须大于0", nameof(numHeads));

        if (embedDim % numHeads != 0) {
            var suggestedEmbedDim = (embedDim / numHeads + 1) * numHeads;
            throw new ArgumentException(
                $"嵌入维度({embedDim})必须能被注意力头数({numHeads})整除。" +
                $"建议使用的嵌入维度: {suggestedEmbedDim}",
                nameof(embedDim));
        }

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
