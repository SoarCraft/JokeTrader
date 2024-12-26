namespace JokeTrader.Torch;

using Microsoft.EntityFrameworkCore;
using TorchSharp;

internal class JokerDataEnumerator : IAsyncEnumerator<(torch.Tensor, torch.Tensor)> {
    private JokerContext context { get; }

    private JokerOption option { get; }

    private int viewSize { get; }

    private int windowSize { get; }

    private ILogger<JokerDataEnumerator> logger { get; }

    private DateTime currentStartTime { get; set; }

    private List<SeriesDataRow>? currentBatchData { get; set; }

    private int currentIndex { get; set; }

    public JokerDataEnumerator(JokerContext context, JokerOption option,
        int viewSize, int windowSize, ILogger<JokerDataEnumerator> logger) {
        this.context = context;
        this.option = option;
        this.viewSize = viewSize;
        this.windowSize = windowSize;
        this.logger = logger;

        this.currentStartTime = this.option.HistoryStart;
        this.currentIndex = -1;
    }

    public async Task LoadNextBatch() {
        var queryEndTime = this.currentStartTime.AddMinutes(this.viewSize * this.option.BatchSize);
        var interval = (int)this.option.KlineInterval / 60;

        var rawData = await this.context.BTCKLines
            .OrderBy(k => k.StartTime)
            .Where(k => k.StartTime >= this.currentStartTime && k.StartTime < queryEndTime)
            .Select(k => new {
                KLine = k,
                Ratio = this.context.BTCRatios
                    .Where(r => r.Timestamp == k.StartTime)
                    .Select(r => new { r.BuyRatio, r.SellRatio })
                    .First(),
                Interest = this.context.BTCInterests
                    .Where(i => i.Timestamp == k.StartTime)
                    .Select(i => i.OpenInterest)
                    .First(),
                Fund = this.context.BTCFundRates
                    .OrderByDescending(f => f.Timestamp)
                    .Where(f => f.Timestamp <= k.StartTime)
                    .Select(f => f.FundingRate)
                    .First()
            })
            .Select(x => new SeriesDataRow {
                Timestamp = x.KLine.StartTime,
                OpenPrice = x.KLine.OpenPrice,
                Volume = x.KLine.Volume,
                BuyRatio = x.Ratio.BuyRatio,
                SellRatio = x.Ratio.SellRatio,
                OpenInterest = x.Interest,
                FundingRate = x.Fund,
                Interval = interval
            }).ToListAsync();

        if (this.viewSize == interval) {
            this.currentBatchData = rawData;
            this.logger.LogInformation($"Loaded {rawData.Count} rows from {this.currentStartTime} to {queryEndTime}");
            return;
        }

        var aggregated = new List<SeriesDataRow>();
        var skip = this.viewSize / interval;

        for (var i = 0; i < rawData.Count; i += skip) {
            var slice = rawData.Skip(i).Take(skip).ToList();

            if (!slice.Any())
                break;

            aggregated.Add(new() {
                Timestamp = slice.First().Timestamp,
                OpenPrice = slice.Average(x => x.OpenPrice),
                Volume = slice.Sum(x => x.Volume),
                BuyRatio = slice.Average(x => x.BuyRatio),
                SellRatio = slice.Average(x => x.SellRatio),
                FundingRate = slice.Average(x => x.FundingRate),
                OpenInterest = slice.Average(x => x.OpenInterest),
                Interval = this.viewSize
            });
        }

        this.currentBatchData = aggregated; 
        this.logger.LogInformation($"Aggregated {aggregated.Count} rows from {this.currentStartTime} to {queryEndTime} with {this.viewSize} viewSize");
    }

    public ValueTask DisposeAsync() => throw new NotImplementedException();

    public ValueTask<bool> MoveNextAsync() => throw new NotImplementedException();

    public (torch.Tensor, torch.Tensor) Current { get; }
}
