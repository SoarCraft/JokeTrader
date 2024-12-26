namespace JokeTrader.Torch;

using System.Collections;
using TorchSharp;

internal class JokerDataEnumerator(JokerContext context, JokerOption option, int viewSize, int windowSize,
    ILogger<JokerDataEnumerator> logger) : IEnumerator<(torch.Tensor, torch.Tensor)> {

    public bool MoveNext() => throw new NotImplementedException();

    public void Reset() => throw new NotImplementedException();

    public (torch.Tensor, torch.Tensor) Current { get; }

    object IEnumerator.Current => this.Current;

    public void Dispose() => throw new NotImplementedException();
}
