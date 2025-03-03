// Copyright (c) 2012-2022 Sandvik Coromant(r). All rights reserved.
// Created by Sandvik Coromant Trondheim
// 2021-1-14

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Localization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;


// DefaultCookieName => .AspNetCore.Culture
// Content example: c%3Dde%7Cuic%3Dde
// de ui de

namespace MVP.Controllers
{
    [Route("[controller]/[action]")]
    public class CultureController : Controller
    {

        private readonly IOptions<RequestLocalizationOptions> _locOptions;

        public CultureController(IOptions<RequestLocalizationOptions> LocOptions)
        {
            _locOptions = LocOptions;

            //  try stopping fallback to another language
            //_locOptions.Value.FallBackToParentCultures = false;
            //_locOptions.Value.FallBackToParentUICultures = false;


        }

        public IActionResult SetCulture(string culture, string redirectUri)
        {
            if (culture != null)
            {
                HttpContext.Response.Cookies.Append(
                    CookieRequestCultureProvider.DefaultCookieName,
                    CookieRequestCultureProvider.MakeCookieValue(
                        new RequestCulture(culture)));

                _locOptions.Value.DefaultRequestCulture = new RequestCulture(culture);

                // Try to change the default
                //System.Globalization.CultureInfo.CurrentCulture = _locOptions.Value.DefaultRequestCulture.Culture;
                //System.Globalization.CultureInfo.CurrentUICulture = _locOptions.Value.DefaultRequestCulture.UICulture;

            }

            return LocalRedirect(redirectUri);
        }
    }
}


