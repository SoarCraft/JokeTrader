namespace JokeTrader.Torch;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class TransformerBlock : Module<Tensor, Tensor> {
    private readonly FlashMultiHeadAttention attention;

    private readonly Sequential feedForward;

    private readonly LayerNorm norm1;

    private readonly LayerNorm norm2;

    private readonly Dropout dropout;

    public TransformerBlock(int embedDim, int numHeads, float dropoutRate) : base(nameof(TransformerBlock)) {
        this.attention = new(embedDim, numHeads, dropoutRate);

        this.feedForward = Sequential(
            Linear(embedDim, embedDim * 4),
            ReLU(),
            Linear(embedDim * 4, embedDim)
        );

        this.norm1 = LayerNorm(embedDim);
        this.norm2 = LayerNorm(embedDim);
        this.dropout = Dropout(dropoutRate);

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input) {
        using var _ = NewDisposeScope();

        var attnOutput = this.attention.forward(this.norm1.forward(input));
        input += this.dropout.forward(attnOutput);

        var ffnOutput = this.feedForward.forward(this.norm2.forward(input));
        return (input + this.dropout.forward(ffnOutput)).MoveToOuterDisposeScope();
    }
}
