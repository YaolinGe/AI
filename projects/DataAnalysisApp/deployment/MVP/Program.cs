using MVP.Controllers;
using MVP.Data;
using MVP.Localization;
using MVP.Shared;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Mvc;
using Sandvik.Coromant.Common.Localization;
using Sandvik.Coromant.Common.Localization;
using Sandvik.Coromant.CoroPlus.Blazor.Components.Localization;
using Sandvik.Coromant.CoroPlus.Blazor.Components.Markdown;
using Sandvik.Coromant.CoroPlus.Blazor.Components.Navigation;
using Sandvik.Coromant.CoroPlus.Tooling.SilentTools.BlazorApp.Shared;
using System.Globalization;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddLocalization();
//var _ = AppCulture.PcCulture;
builder.Services.AddScoped<IAppCulture, AppCulture>();
builder.Services.AddScoped<ILocalizer, LanguageResource<Resource>>();

// Add services to the container.
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
//builder.Services.AddSingleton<WeatherForecastService>();
builder.Services.AddScoped<ILocalizer, LanguageResource<Resource>>();
builder.Services.AddScoped<IActivePageProvider, ActivePageProvider>();
builder.Services.AddScoped<NavigationControl>();
builder.Services.AddScoped<IFindMarkdown, FindMarkdown>();

var supportedCultures = new List<CultureInfo> {
                   new CultureInfo("en-US"),
                   new CultureInfo("en-GB"),
                   new CultureInfo("de-DE"),
                   new CultureInfo("es-ES"),
                   new CultureInfo("fr-FR"),
                   new CultureInfo("it-IT"),
                   new CultureInfo("ja-JP"),
                   new CultureInfo("ko-KR"),
                   new CultureInfo("pl-PL"),
                   new CultureInfo("pt-PT"),
               };

builder.Services.AddRequestLocalization(options =>
{
    if (supportedCultures.Contains(CultureInfo.CurrentUICulture))
    {
        options.SetDefaultCulture(CultureInfo.CurrentUICulture.Name);
    }
    else
    {
        options.SetDefaultCulture("en-US");
    }
    options.SupportedCultures = supportedCultures;
    options.SupportedUICultures = supportedCultures;

    // Insert(0, or 1 if we do not want to respect cookie for language (suspect cookie is #1 )
    options.RequestCultureProviders.Insert(2, new UserCustomRequestCultureProvider());

});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

app.Run();
