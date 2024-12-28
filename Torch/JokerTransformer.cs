namespace JokeTrader.Torch;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class JokerTransformer : Module<Tensor, Tensor> {
    private readonly Sequential inputProjection;

    private readonly Sequential outputHead;

    private readonly PositionalEncoding posEncoding;

    private readonly TransformerEncoder transformer;

    public JokerTransformer(int featureDim, int embedDim, int numHeads,
        int numLayers, float dropoutRate = 0.1f) : base(nameof(JokerTransformer)) {
        this.inputProjection = Sequential(
            Linear(featureDim - 1, embedDim),
            ReLU(),
            Dropout(dropoutRate)
        );

        this.posEncoding = new();

        var encoderLayer = TransformerEncoderLayer(
            embedDim,
            numHeads,
            embedDim * 4,
            dropoutRate
        );

        this.transformer = TransformerEncoder(
            encoderLayer,
            numLayers
        );

        this.outputHead = Sequential(
            Linear(embedDim, embedDim / 2),
            ReLU(),
            Dropout(dropoutRate),
            Linear(embedDim / 2, 2)
        );

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input) {
        using var _ = NewDisposeScope();

        input = this.posEncoding.forward(input);
        input = this.inputProjection.forward(input);

        input = this.transformer.forward(input, null, null);

        var attentionWeights = softmax(matmul(input, input.transpose(-2, -1)), -1);

        var globalFeatures = matmul(attentionWeights, input).mean([1]);

        return this.outputHead.forward(globalFeatures).MoveToOuterDisposeScope();
    }
}
