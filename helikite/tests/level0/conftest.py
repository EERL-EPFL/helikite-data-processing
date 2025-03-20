import pytest
import shutil
import os


@pytest.fixture
def campaign_data(tmpdir):
    # Use absolute path for the source directory
    source_dir = os.path.join(
        os.path.dirname(__file__), "..", "resources", "campaigns"
    )

    # Verify the path
    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"Source directory {source_dir} does not exist"
        )

    # Copy contents to tmpdir
    temp_campaign_dir = tmpdir.mkdir("campaign_data")
    shutil.copytree(source_dir, str(temp_campaign_dir), dirs_exist_ok=True)

    return temp_campaign_dir
