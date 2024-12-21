import json
import random
import os
import pandas as pd
from tqdm import tqdm
from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

EMBEDDING_PROMPTS = {
    "clinicaltrials.gov": "Given a patient record, retrieve eligible clinical trials",
}


class ClinicalTrialsData(Dataset):
    """
    A dataset class for retrieving patient-trial pairs and constructing training samples.
    
    Expects the following files in `file_path` directory:
      - patients.json: A JSON file with patient records indexed by some ID.
      - trials.json: A JSON file with trial descriptions indexed by some ID.
      - trial_pairs.csv: A CSV file with columns [query-id, positive, negative] indicating
        which patient (query) maps to which positive and negative trial candidates.
    
    The class will:
      - Load the data
      - Create a list of DataSample objects
      - Optionally shuffle and batch the data
      - On training split, return TrainSample with [query, positive, negative]
      - On validation split, return DataSample (or raise error if not supported)
    """

    def __init__(
        self,
        dataset_name: str = "clinicaltrials.gov",
        split: str = "validation",
        file_path: str = "/home/than/DeepLearning/llm2vec/cache/",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self._load_data(file_path)

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path: str):
        logger.info(f"Loading clinical trial data from {file_path}...")

        # Check files existence
        trials_path = os.path.join(file_path, "trials.json")
        patients_path = os.path.join(file_path, "patients.json")
        pairs_path = os.path.join(file_path, "trial_pairs.csv")

        with open(trials_path, "r") as f:
            trials = json.load(f)
        with open(patients_path, "r") as f:
            patients = json.load(f)
        df_pairs = pd.read_csv(pairs_path)

        data_map = {self.dataset_name: []}
        all_samples = []
        id_ = 0

        dataset = self.dataset_name
        instruction = EMBEDDING_PROMPTS[dataset]
        logger.info(f"Loading dataset {dataset}...")

        # Each row in df_pairs corresponds to a patient and their positive/negative trials
        for i, pair in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
            query_id = pair["query-id"]
            pos_id = pair["positive"]
            neg_id = pair["negative"]

            # Construct the strings
            query = f"{instruction}; {self.separator}{patients[query_id]}"
            pos = f"{self.separator}{trials[pos_id]}"
            neg = f"{self.separator}{trials[neg_id]}"

            data_map[dataset].append(id_)
            all_samples.append(
                DataSample(
                    id_=id_,
                    query=query,
                    positive=pos,
                    negative=neg,
                    task_name=dataset,
                )
            )
            id_ += 1

        # At this point, we have all_samples and data_map filled
        logger.info(f"Loaded {len(all_samples)} raw samples for dataset {dataset}.")

        # Shuffle data if required
        if self.shuffle_individual_datasets:
            random.shuffle(data_map[dataset])

        # Batching
        logger.info(
            f"Batching data for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        dataset_samples = data_map[dataset]
        for i in range(0, len(dataset_samples), self.effective_batch_size):
            batch = dataset_samples[i : i + self.effective_batch_size]
            if len(batch) == self.effective_batch_size:
                all_batches.append(batch)
            else:
                logger.info(f"Skipping incomplete batch of size {len(batch)} at the end.")

        # Shuffle the batches again if desired
        random.shuffle(all_batches)

        final_idx_order = [idx for batch in all_batches for idx in batch]
        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            # During training, return TrainSample with query, positive, negative
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative],
                label=1.0
            )
        elif self.split == "validation":
            assert False, "ClinicalTrial data does not have a validation split."
