﻿namespace JokeTrader.Torch;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

internal class JokerTransformer : Module<Tensor, Tensor> {
    private readonly Sequential inputProjection;

    private readonly ModuleDict<Sequential> outputHead;

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

        this.outputHead = ModuleDict(
            ("direction", Sequential(Linear(embedDim, 1), Sigmoid())),
            ("magnitude", Sequential(Linear(embedDim, 1), ReLU()))
        );

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

        var direction = this.outputHead["direction"].forward(globalFeatures);
        var magnitude = this.outputHead["magnitude"].forward(globalFeatures);

        return cat([direction, magnitude], dim: 1).MoveToOuterDisposeScope();
    }
}
