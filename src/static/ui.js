// src/static/ui.js

/**
 * Handles the navigation for the multi-page dashboard.
 * It shows/hides page content based on the active sidebar link.
 */
document.addEventListener('DOMContentLoaded', () => {
    // Get all sidebar links and page content elements
    const sidebarLinks = document.querySelectorAll('.sidebar-menu a');
    const pageContents = document.querySelectorAll('.page-content');

    /**
     * Shows a specific page and hides all others.
     * @param {string} pageId The ID of the page to show (e.g., 'dashboard-page').
     */
    function showPage(pageId) {
        // Hide all pages
        pageContents.forEach(page => {
            page.classList.remove('active');
        });
        
        // Show the selected page
        const activePage = document.getElementById(pageId);
        if (activePage) {
            activePage.classList.add('active');
        }
    }

    // Add a click listener to each sidebar link
    sidebarLinks.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault(); // Prevent default link behavior
            
            // Remove 'active' class from all links
            sidebarLinks.forEach(item => {
                item.classList.remove('active');
            });

            // Add 'active' class to the clicked link
            link.classList.add('active');

            // Get the data-page attribute to determine which page to show
            const pageName = link.getAttribute('data-page');
            const pageId = `${pageName}-page`;
            
            // Show the corresponding page content
            showPage(pageId);
        });
    });

    // On initial load, ensure the correct page is shown
    // This is optional but good practice
    const initialPageId = document.querySelector('.sidebar-menu a.active').getAttribute('data-page') + '-page';
    showPage(initialPageId);
});
