using JokeTrader;
using JokeTrader.Services;
using Microsoft.EntityFrameworkCore;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddOptions<JokerOption>();

builder.Services.AddBybit();

builder.Services.AddDbContext<JokerContext>(
    options => options.UseSqlServer(
            builder.Configuration.GetConnectionString("DefaultConnection"))
        .EnableDetailedErrors()
        .EnableSensitiveDataLogging());

builder.Services.AddTransient<SymbolService>();
builder.Services.AddHostedService<KLineService>();
builder.Services.AddHostedService<RatioService>();

var app = builder.Build();

using (var scope = app.Services.CreateScope()) {
    var services = scope.ServiceProvider;
    var symbol = services.GetRequiredService<SymbolService>();
    await symbol.Prepare();
}

await app.RunAsync();
