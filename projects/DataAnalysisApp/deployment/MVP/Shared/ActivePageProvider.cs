using Sandvik.Coromant.CoroPlus.Blazor.Components.Navigation;

namespace MVP.Shared
{
    internal class ActivePageProvider : IActivePageProvider
    {
        public ActivePageProvider()
        {
            ChangeActivePages();
        }

        private readonly List<INavMenuItem> topMenuItems = new();
        private readonly List<INavMenuItem> bottomMenuItems = new();

        public void ChangeActivePages(string navigateTo = "")
        {
            // Top
            topMenuItems.Clear();
            topMenuItems.Add(new NavMenuLink("Home", "home", string.Empty));
            topMenuItems.Add(new NavMenuLink("Counter", "add", "counter"));
            topMenuItems.Add(new NavMenuLink("Fetch data", "analytics", "fetchdata"));

            // Bottom
            bottomMenuItems.Clear();
            bottomMenuItems.Add(new NavMenuLink("PageTitle_Settings", "settings", "settings"));

            var helpSubItems = new List<INavMenuItem>()
            {
                new NavMenuLink("Home", string.Empty, "help/home"),
                new NavMenuLink("Counter", string.Empty, "help/counter"),
                new NavMenuLink("Weather", string.Empty, "help/weather"),
                new NavMenuLink("Settings", string.Empty, "help/settings"),
            };
            helpSubItems.Add(new NavMenuSeparator());
            helpSubItems.Add(new NavMenuLink("Silent Tools™ Plus", string.Empty, "https://www.sandvik.coromant.com/en-us/tools/turning-tools/internal-turning-tools/silent-tools-turning/silent-tools-plus/get-started-silent-tools-plus"));

            bottomMenuItems.Add(new NavMenuGroup("Settings_WidgetHelp.Title", "help", NavMenuGroupType.SubItems, helpSubItems.ToArray()));

            OnActivePagesChanged(navigateTo);
        }

        public event EventHandler<ActivePagesChangedEventArgs> ActivePagesChanged;

        private void OnActivePagesChanged(string navigateTo)
        {
            ActivePagesChanged?.Invoke(this, new ActivePagesChangedEventArgs() { NavigateTo = navigateTo });
        }



        public IEnumerable<INavMenuItem> GetTopMenuItems()
        {
            return topMenuItems;
        }

        public IEnumerable<INavMenuItem> GetBottomMenuItems()
        {
            return bottomMenuItems;
        }
    }
}
