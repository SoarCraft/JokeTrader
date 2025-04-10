﻿namespace JokeTrader.Torch;

using Microsoft.EntityFrameworkCore;
using TorchSharp;

internal class JokerDataEnumerator : IAsyncEnumerator<(torch.Tensor, torch.Tensor)> {
    public JokerDataEnumerator(JokerContext context, JokerOption option,
        int viewSize, int windowSize, ILogger<JokerDataEnumerator> logger) {
        this.context = context;
        this.option = option;
        this.viewSize = viewSize;
        this.windowSize = windowSize;
        this.logger = logger;

        this.currentStartTime = this.option.HistoryStart;
    }

    private JokerContext context { get; }

    private JokerOption option { get; }

    private int viewSize { get; }

    private int windowSize { get; }

    private ILogger<JokerDataEnumerator> logger { get; }

    private DateTime currentStartTime { get; set; }

    private List<SeriesDataRow>? currentBatchData { get; set; }

    public ValueTask DisposeAsync() {
        this.currentBatchData?.Clear();
        this.logger.LogDebug("Dispose current batch data");
        return ValueTask.CompletedTask;
    }

    public async ValueTask<bool> MoveNextAsync() {
        if (this.currentBatchData == null)
            return await this.LoadNextBatch();

        this.currentStartTime = this.currentBatchData[^1].Timestamp;

        return await this.LoadNextBatch();
    }

    public (torch.Tensor, torch.Tensor) Current {
        get {
            if (this.currentBatchData is null)
                throw new InvalidOperationException("Invalid current batch data");

            var availableBatch = this.currentBatchData.Count - this.windowSize + 1;
            if (this.option.BatchSize > availableBatch)
                this.logger.LogInformation("Not enough data for batch, reduced this batch size to {0}", availableBatch);

            var featDim = this.currentBatchData.First().ToArray().Length;
            var input = torch.zeros([
                availableBatch,
                this.windowSize - 1,
                featDim
            ], torch.ScalarType.Float32);

            var target = torch.zeros([
                availableBatch,
                2
            ], torch.ScalarType.Float32);

            for (var batch = 0; batch < availableBatch; batch++) {
                for (var window = 0; window < this.windowSize - 1; window++) {
                    var dataIndex = batch + window;
                    var features = this.currentBatchData[dataIndex].ToArray();
                    var featureTensor = torch.tensor(features, torch.ScalarType.Float32);
                    input[batch, window] = featureTensor;
                }

                var lastIndex = batch + this.windowSize - 2;

                var lastPrice = this.currentBatchData[lastIndex].OpenPrice;
                var nextPrice = this.currentBatchData[lastIndex + 1].OpenPrice;

                var priceChange = (nextPrice - lastPrice) / lastPrice;
                target[batch, 0] = priceChange > 0 ? 1 : 0;
                target[batch, 1] = Math.Abs(priceChange);
            }

            return (input.to(this.option.Device), target.to(this.option.Device));
        }
    }

    public async Task<bool> LoadNextBatch() {
        this.currentBatchData?.Clear();

        var requiredTimeSteps = this.windowSize + this.option.BatchSize - 1;
        var queryEndTime = this.currentStartTime.AddMinutes(this.viewSize * requiredTimeSteps);

        var interval = (int)this.option.KlineInterval / 60;
        var requiredDataNum = requiredTimeSteps * this.viewSize / interval;

        var rawData = await this.context.KLines
            .Where(k => k.StartTime >= this.currentStartTime && k.StartTime < queryEndTime)
            .Select(k => new {
                KLine = k,
                Ratio = this.context.Ratios
                    .Where(r => r.Timestamp == k.StartTime)
                    .Select(r => new { r.BuyRatio, r.SellRatio })
                    .First(),
                Interest = this.context.Interests
                    .Where(i => i.Timestamp == k.StartTime)
                    .Select(i => i.OpenInterest)
                    .First(),
                Fund = this.context.FundRates
                    .OrderByDescending(f => f.Timestamp)
                    .Where(f => f.Timestamp <= k.StartTime)
                    .Select(f => f.FundingRate)
                    .First()
            })
            .Select(x => new SeriesDataRow {
                Timestamp = x.KLine.StartTime,
                OpenPrice = x.KLine.OpenPrice,
                HighPrice = x.KLine.HighPrice,
                LowPrice = x.KLine.LowPrice,
                ClosePrice = x.KLine.ClosePrice,
                Volume = x.KLine.Volume,
                BuyRatio = x.Ratio.BuyRatio,
                SellRatio = x.Ratio.SellRatio,
                OpenInterest = x.Interest,
                FundingRate = x.Fund,
                Interval = interval
            }).ToListAsync();

        this.currentBatchData = this.aggregateData(rawData);

        this.logger.LogDebug(
            $"Load {rawData.Count} rows required {requiredDataNum} to {this.currentBatchData.Count} " +
            $"from {this.currentStartTime} to {queryEndTime} with {this.viewSize} viewSize");

        if (this.currentBatchData.Count >= this.windowSize)
            return true;

        this.logger.LogInformation("Not enough data for next one batch");
        return false;
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
}
