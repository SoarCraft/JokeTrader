using JokeTrader;
using Microsoft.EntityFrameworkCore;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddBybit();

builder.Services.AddDbContext<JokerContext>(
    options => options.UseSqlServer(
            builder.Configuration.GetConnectionString("DefaultConnection"))
        .EnableDetailedErrors()
        .EnableSensitiveDataLogging());

builder.Services.AddHostedService<KLineService>();

var app = builder.Build();

await app.RunAsync();
