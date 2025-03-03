// Copyright (c) 2012-2022 Sandvik Coromant(r). All rights reserved.
// Created by Sandvik Coromant Trondheim
// 2021-1-14

using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Localization;

namespace MVP.Controllers
{
    public class UserCustomRequestCultureProvider : RequestCultureProvider
    {
        public string Culture { get; set; }

        // This will get called on every request to the server

        public override Task<ProviderCultureResult> DetermineProviderCultureResult(HttpContext httpContext)
        {
            if (httpContext == null)
            {
                throw new ArgumentNullException(nameof(httpContext));
            }

            if (string.IsNullOrEmpty(Culture))
            {
                return Task.FromResult((ProviderCultureResult)null);
            }

            return Task.FromResult(new ProviderCultureResult(Culture));
        }

    }
}

