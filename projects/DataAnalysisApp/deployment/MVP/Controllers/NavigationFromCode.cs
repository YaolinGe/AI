using Microsoft.AspNetCore.Components;

namespace MVP.Controllers
{
    public class NavigationFromCode
    {
        public NavigationManager? NavigationManager { get; set; }

        private CancellationTokenSource? cts;

        public void RequestNavigation(string url, bool force = false)
        {
            if (!TryNavigate(url, force))
            {
                cts?.Cancel();
                cts?.Dispose();

                cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                Task.Run(async () => await HandleNavigationRequestAsync(url, force, cts.Token));
            }
        }

        private bool TryNavigate(string url, bool force)
        {
            if (NavigationManager != null)
            {
                NavigationManager.NavigateTo(url, forceLoad: force);
                return true;
            }

            return false;
        }

        private async Task HandleNavigationRequestAsync(string url, bool force, CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(300, CancellationToken.None);
                if (TryNavigate(url, force))
                {
                    cts?.Cancel();
                    cts?.Dispose();
                    cts = null;
                    return;
                }
            }
        }
    }
}
