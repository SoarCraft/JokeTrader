using Microsoft.Extensions.Hosting;

var host = Host.CreateApplicationBuilder(args);

var app = host.Build();

await app.RunAsync();
