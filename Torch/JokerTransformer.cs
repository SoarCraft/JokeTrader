namespace JokeTrader.Torch;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class JokerTransformer : Module<Tensor, Tensor> {
    private readonly Sequential inputProjection;

    private readonly Sequential outputHead;

    private readonly PositionalEncoding posEncoding;

    private readonly ModuleList<TransformerBlock> transformerBlocks;

    public JokerTransformer(int featureDim, int embedDim, int numHeads,
        int numLayers, float dropoutRate = 0.1f) : base(nameof(JokerTransformer)) {
        this.inputProjection = Sequential(
            Linear(featureDim - 1, embedDim),
            ReLU(),
            Dropout(dropoutRate)
        );

        this.posEncoding = new();

        this.transformerBlocks = new();
        for (var i = 0; i < numLayers; i++)
            this.transformerBlocks.Add(new(embedDim, numHeads, dropoutRate));

        this.outputHead = Sequential(
            Linear(embedDim * 2, embedDim),
            ReLU(),
            Dropout(dropoutRate),
            Linear(embedDim, 2)
        );

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input) {
        using var _ = NewDisposeScope();

        input = this.inputProjection.forward(input);
        input = this.posEncoding.forward(input);

        input = this.transformerBlocks
            .Aggregate(input, (current, block) => block.forward(current));

        var historyFeatures = input[.., ..^1, ..];
        var lastFeatures = input[.., -1, ..];

        var attention = matmul(
            historyFeatures,
            lastFeatures.unsqueeze(-1)
        );

        var attentionWeights = softmax(attention, 1);

        var globalFeatures = (attentionWeights * historyFeatures).sum(1);

        input = cat([globalFeatures, lastFeatures], -1);

        return this.outputHead.forward(input).MoveToOuterDisposeScope();
    }
}
