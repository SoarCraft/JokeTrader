using JokeTrader;
using JokeTrader.Services;
using JokeTrader.Torch;
using Microsoft.EntityFrameworkCore;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddOptions<JokerOption>();

builder.Services.AddBybit();

builder.Services.AddDbContextFactory<JokerContext>(
    options => options.UseSqlServer(
            builder.Configuration.GetConnectionString("DefaultConnection"))
        .EnableDetailedErrors()
        .EnableSensitiveDataLogging());

builder.Services.AddTransient<SymbolService>();
builder.Services.AddTransient<KLineService>();
builder.Services.AddTransient<RatioService>();
builder.Services.AddTransient<FundRateService>();
builder.Services.AddTransient<InterestService>();

builder.Services.AddSingleton<JokerDataLoader>();
builder.Services.AddHostedService<Orchestration>();

var app = builder.Build();

await app.RunAsync();
