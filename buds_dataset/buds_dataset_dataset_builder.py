from typing import Iterator, Tuple, Any

import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class BudsDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(24,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '1x gripper position, 16x robot end-effector homogeneous matrix].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x end effector delta pose, '
                            '1x gripper position].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples('buds_real_kitchen_demo.hdf5'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, dataset_path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        dataset = h5py.File(dataset_path, "r")

        def _parse_example(dataset, ep_idx):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            data = dataset[f"data/demo_{ep_idx}"]
            episode = []
            language_instruction = "Take the lid off the pot, put the pot on the plate, and use the tool to push to pot to the front of the table."
            for i in range(data.attrs["num_samples"]):
                # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                state = np.concatenate([
                            data["joint_states"][()][i],
                            data["gripper_states"][()][i],
                            data["ee_states"][()][i],
                        ])
                episode.append({
                    'observation': {
                        'image': data["agentview_rgb"][()][i],
                        'wrist_image': data["eye_in_hand_rgb"][()][i],
                        'state': state,
                    },
                    'action': data["actions"][()][i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (data.attrs["num_samples"] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (data.attrs["num_samples"] - 1),
                    'is_terminal': i == (data.attrs["num_samples"] - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': f"{ep_idx}"
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return f"{ep_idx}", sample

        for ep_idx in range(50):
            yield _parse_example(dataset, ep_idx)

        # # create list of all examples
        # episode_paths = glob.glob(path)

        # # for smallish datasets, use single-thread parsing
        # for sample in episode_paths:
        #     yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

