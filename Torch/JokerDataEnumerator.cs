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
        this.currentBatchData?.Clear();

        var requiredTimeSteps = this.windowSize + this.option.BatchSize - 1;
        var queryEndTime = this.currentStartTime.AddMinutes(this.viewSize * requiredTimeSteps);
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

        var aggregate = this.aggregateData(rawData);
        this.currentBatchData = await this.normalizeData(aggregate);

        this.logger.LogInformation($"Load {this.currentBatchData.Count} rows from {this.currentStartTime} to {queryEndTime} with {this.viewSize} viewSize");
    }

    private async Task<List<SeriesDataRow>> normalizeData(List<SeriesDataRow> rawData) {
        var normalization = await this.context.Normalizations
            .Where(n => n.SymbolId == this.option.Symbol)
            .Select(x => new { x.Feature, x.Mean, x.Std })
            .ToDictionaryAsync(k => k.Feature, v => (v.Mean, v.Std));

        var volumeMean = rawData.CalculateMean(data => data.Volume);
        var volumeStd = rawData.CalculateStd(data => data.Volume, volumeMean);

        var priceNorm = normalization[nameof(SeriesFeatures.OpenPrice)];
        var interestNorm = normalization[nameof(SeriesFeatures.OpenInterest)];

        foreach (var data in rawData) {
            data.OpenPrice = priceNorm.Std == 0
                ? 0
                : (data.OpenPrice - priceNorm.Mean) / priceNorm.Std;

            data.OpenInterest = interestNorm.Std == 0
                ? 0
                : (data.OpenInterest - interestNorm.Mean) / interestNorm.Std;

            data.Volume = volumeStd == 0
                ? 0
                : (data.Volume - volumeMean) / volumeStd;
            
            data.Interval /= Math.Max(1, JokerDataLoader.ViewSizes.Max());
        }

        return rawData;
    }

    private List<SeriesDataRow> aggregateData(List<SeriesDataRow> rawData) {
        if (this.viewSize == rawData.First().Interval)
            return rawData;

        var aggregated = rawData
            .Chunk(this.viewSize / rawData.First().Interval)
            .Select(slice => new SeriesDataRow {
                Timestamp = slice.First().Timestamp,
                OpenPrice = slice.Average(x => x.OpenPrice),
                Volume = slice.Sum(x => x.Volume),
                BuyRatio = slice.Average(x => x.BuyRatio),
                SellRatio = slice.Average(x => x.SellRatio),
                FundingRate = slice.Average(x => x.FundingRate),
                OpenInterest = slice.Average(x => x.OpenInterest),
                Interval = this.viewSize
            }).ToList();

        return aggregated;
    }

    public async ValueTask DisposeAsync() {
        this.currentBatchData?.Clear();
    }

    public async ValueTask<bool> MoveNextAsync() {
        if (this.currentBatchData == null) {
            await this.LoadNextBatch();
            this.currentIndex = 0;
            return true;
        }

        if (this.currentIndex + this.windowSize + this.option.BatchSize >= this.currentBatchData.Count) {
            this.currentStartTime = this.currentBatchData[this.currentIndex].Timestamp;
            await this.LoadNextBatch();
            this.currentIndex = 0;
        } else
            this.currentIndex++;

        return this.currentBatchData is not null &&
               this.currentBatchData.Count > this.windowSize + this.option.BatchSize;
    }

    public (torch.Tensor, torch.Tensor) Current {
        get {
            if (this.currentBatchData is null || this.currentIndex < 0)
                throw new InvalidOperationException("Invalid current batch data");

            var featDim = this.currentBatchData.First().ToArray().Length;
            var input = torch.zeros([
                this.option.BatchSize,
                this.windowSize,
                featDim
            ], torch.ScalarType.Float32, this.option.Device);

            var target = torch.zeros([
                this.option.BatchSize,
                2
            ], torch.ScalarType.Float32, this.option.Device);

            for (var batch = 0; batch < this.option.BatchSize; batch++) {
                for (var window = 0; window < this.windowSize; window++) {
                    var dataIndex = this.currentIndex + batch + window;
                    var features = this.currentBatchData[dataIndex].ToArray();

                    for (var feat = 0; feat < featDim; feat++)
                        input[batch, window, feat] = features[feat];
                }

                var lastIndex = this.currentIndex + batch + this.windowSize - 1;

                var lastPrice = this.currentBatchData[lastIndex].OpenPrice;
                var nextPrice = this.currentBatchData[lastIndex + 1].OpenPrice;

                var priceChange = (nextPrice - lastPrice) / lastPrice;
                target[batch, 0] = priceChange > 0 ? 1 : 0;
                target[batch, 1] = Math.Abs(priceChange);
            }

            return (input, target);
        }
    }
}
