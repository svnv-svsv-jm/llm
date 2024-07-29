__all__ = ["download_pdfs_on_page"]

from loguru import logger
import typing as ty

from selenium.webdriver.common.by import By

from .driver import DRIVER_TYPE


def download_pdfs_on_page(web_driver: DRIVER_TYPE) -> bool:
    """Downloads all PDF's in the web page.

    Args:
        web_driver (DRIVER_TYPE):
            Current web driver.

    Returns:
        bool: Whether the download button has been clicked at least once or not.
    """
    # Locate the table element
    tables = web_driver.find_elements(By.XPATH, "//table")

    has_clicked = False
    for table in tables:
        # Find all <tr> elements within the table
        rows = table.find_elements(By.TAG_NAME, "tr")

        # Iterate over each <tr> element
        for row in rows:
            # Print the row or process it as needed
            logger.info(f"Row: {row}")
            logger.info(row.get_attribute("outerHTML"))

            # Optionally, find all <td> elements within this row
            cells = row.find_elements(By.TAG_NAME, "td")
            for cell in cells:
                # Print the outerHTML of the cell
                cell_html = cell.get_attribute("outerHTML")
                logger.info(cell_html)
                # Check if the cell contains the <a> element with the specific attributes
                try:
                    download_button = cell.find_element(
                        By.XPATH, ".//a[@role='button' and contains(@title, 'Scarica il pdf')]"
                    )
                    download_button.click()  # Click
                    has_clicked = True
                except:
                    # If the specific <a> element is not found, continue to the next cell
                    continue

    # Return
    return has_clicked
