namespace JokeTrader.Torch;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class JokerTransformer : Module<Tensor, Tensor> {
    private readonly Sequential inputProjection;

    private readonly Sequential direction;

    private readonly Sequential magnitude;

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

        this.direction = Sequential(Linear(embedDim, 1), Sigmoid());

        this.magnitude = Sequential(Linear(embedDim, 1), ReLU());

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input) {
        using var _ = NewDisposeScope();

        input = this.posEncoding.forward(input);
        input = this.inputProjection.forward(input);

        input = this.transformer.forward(input, null, null);

        var seqLen = input.shape[1];
        var weights = linspace(1.0, 0.1, seqLen, device: input.device).unsqueeze(0).unsqueeze(-1);

        input *= weights;
        var globalFeatures = input.sum(dim: 1);

        var dir = this.direction.forward(globalFeatures);
        var mag = this.magnitude.forward(globalFeatures);

        return cat([dir, mag], dim: 1).MoveToOuterDisposeScope();
    }
}
