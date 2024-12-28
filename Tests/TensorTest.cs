namespace JokeTrader.Tests;

using Torch;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static TorchSharp.torch;

[TestClass]
public sealed class TensorTest {
    [TestMethod]
    public void TestTransformer() {
        var batchSize = 1;
        var seqLen = 10;
        var featureDim = 5;
        var embedDim = 64;
        var numHeads = 4;
        var numLayers = 2;

        using var input = randn(new long[] { batchSize, seqLen, featureDim }, ScalarType.Float32);

        using var model = new JokerTransformer(
            featureDim,
            embedDim,
            numHeads,
            numLayers
        );

        using var output = model.forward(input);

        Assert.AreEqual(batchSize, output.shape[0]);
        Assert.AreEqual(2, output.shape[1]);

        Console.WriteLine($"Input shape: [{string.Join(", ", input.shape)}]");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.shape)}]");
    }
}
