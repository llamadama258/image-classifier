from omrdatasettools import Downloader, OmrDataset

downloader = Downloader()
downloader.download_and_extract_dataset(OmrDataset.Rebelo1, "data/rebelo1")
downloader.download_and_extract_dataset(OmrDataset.Rebelo2, "data/rebelo2")