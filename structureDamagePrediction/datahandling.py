import os, math
import torch, typing
from torch.utils.data import IterableDataset
from structureDamagePrediction.utils import StartEndLogger
from typing import Tuple

class StructuralDamageDataAndMetadataReader():
    def __get_filenames(self, file_num : int, base_dir = "data/", sensor_base_filename = "data_sensors_case_", 
        sensor_base_file_ext = ".csv", 
        metadata_base_filename = "metaData_case_", 
        metadata_base_file_ext = ".csv",
        l = StartEndLogger()) -> Tuple[str,str] :
        """Returns a tuple of the (sensor_filepath, metadata_filepath)
        """
        
        sensor_filepath = "%s%s%d%s"%(base_dir, sensor_base_filename, file_num, sensor_base_file_ext)
        metadata_filepath = "%s%s%d%s"%(base_dir, metadata_base_filename, file_num, metadata_base_file_ext)

        return (sensor_filepath, metadata_filepath)
    
    def read_data_and_metadata(self, l = StartEndLogger()) -> Tuple[list,list]:
        """Returns a tuple of two lists containing the sequence data and the metadata of the read instances.
        """
        # Init sensor measurements sequence list
        sequence_data = []
        # Init target values list
        sequence_metadata = []

        

        file_cnt = 1
        # While we have both files we need
        sensor_filepath, metadata_filepath = self.__get_filenames(file_cnt)
        instance_list = []
        while os.path.isfile(sensor_filepath) and os.path.isfile(metadata_filepath):
            l.start("Reading data from file #%d"%(file_cnt))
            fdr = FileDataReader(sequence_data_filename=sensor_filepath, meta_data_filename=metadata_filepath)

            # Gather the data
            # and metadata
            data, metadata = fdr.read_data()
            l.end()

            # Add to instance list
            sequence_data.append(data)
            sequence_metadata.append(metadata)

            # DEBUG
            l.log("Data:\n%s"%(str(data)))
            l.log("Metadata:\n%s"%(str(metadata)))

            # Move on
            file_cnt += 1
            sensor_filepath, metadata_filepath = self.__get_filenames(file_cnt)
        
        return sequence_data, sequence_metadata


class BaseDataReader():
    def read_data(self) -> tuple:
        """Returns the data and """
        seq = self.read_sequence()
        meta = self.read_metadata()

        if (seq is not None) and (meta is not None):
            return seq, meta

    def read_sequence(self) -> torch.Tensor:
        return None
    
    def read_metadata(self) -> torch.Tensor:
        return None

class FileDataReader(BaseDataReader):
    def __init__(self, sequence_data_filename: str, meta_data_filename: str):
        self.sequence_filename = sequence_data_filename
        self.metadata_filename = meta_data_filename

    
    def read_sequence(self, ) -> torch.Tensor:
        # Init sequence list
        seq_list = []

        # Read lines
        with open(self.sequence_filename) as sequence_file:
            b_header = False
            # For each line do
            for s_line in sequence_file.readlines():
                # Skip header
                if not b_header:
                    b_header = True
                    continue

                # Ignore empty lines
                if len(s_line.strip()) == 0:
                    continue

                cur_line_fields = s_line.split()
                import torch.types
                cur_line_tensor = torch.tensor(list(map(float,cur_line_fields)), dtype=torch.float)

                seq_list.append(cur_line_tensor)
        
                
        # Convert to tensor and return
        ret_seq = torch.stack(seq_list)
        return ret_seq

    
    def read_metadata(self) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a Tensor containing (as float tensors) the following: the case_id, dmg_perc, dmg_tensor, dmg_loc_x, dmg_loc_y"""

        # Read lines
        with open(self.metadata_filename) as metadata_file:
            # caseStudey Damage_percentage DamageLayer1 DamageLayer2 DamageLayer3 DamageLayer4 DamageLayer5 DamageLocX DamageLocY
            _, mainline1, line1, line2, line3 = metadata_file.readlines() # Load lines, ignoring header
            # Init damages per layer
            dmg_layer_1, dmg_layer_2, dmg_layer_3, dmg_layer_4, dmg_layer_5 = ([], [], [], [], [])
            # Convert and save
            case_id, dmg_perc, dmg11, dmg21, dmg31, dmg41, dmg51, dmg_loc_x, dmg_loc_y = tuple(map(float, mainline1.split()))
            dmg12, _, dmg32, _, dmg52 = tuple(map(float, line1.split()[2:7]))
            dmg13, _, dmg33, _, dmg53 = tuple(map(float, line2.split()[2:7]))
            dmg14, _, dmg34, _, dmg54 = tuple(map(float, line3.split()[2:7]))

            dmg_layer_1.extend([dmg11, dmg12, dmg13, dmg14])
            dmg_layer_2.extend([dmg21, float('nan'), float('nan'), float('nan')])
            dmg_layer_3.extend([dmg31, dmg32, dmg33, dmg34])
            dmg_layer_4.extend([dmg41, float('nan'), float('nan'), float('nan')])
            dmg_layer_5.extend([dmg51, dmg52, dmg53, dmg54])
        
        dmg_tensor = torch.stack(list(map(torch.tensor, [dmg_layer_1, dmg_layer_2, dmg_layer_3, dmg_layer_4, dmg_layer_5])))

        # Init and return value
        ret_metadata = (case_id, torch.tensor(dmg_perc), dmg_tensor, torch.tensor(dmg_loc_x), torch.tensor(dmg_loc_y))
        return ret_metadata

class StructuralDamageDataset(IterableDataset):
    def ___get_info(self, instance):
        res = instance[self.tgt_tuple_index_in_metadata]
        if self.tgt_row_in_metadata is not None:
            res = res[self.tgt_row_in_metadata]
            if self.tgt_col_in_metadata is not None:
                res = res[self.tgt_col_in_metadata]
        return res        
    
    def __init__(self, data_list : list, metadata_list: list, tgt_tuple_index_in_metadata = 1, 
                 tgt_row_in_metadata: int = None , tgt_col_in_metadata: int = None ) -> None:
        super().__init__()
        self.data_list = data_list
        self.metadata_list = metadata_list
        self.tgt_tuple_index_in_metadata = tgt_tuple_index_in_metadata
        self.tgt_row_in_metadata = tgt_row_in_metadata
        self.tgt_col_in_metadata = tgt_col_in_metadata

        # Make sure lengths are the same
        if len(self.data_list) != len(self.metadata_list):
            raise RuntimeError("Data entries are more/less than the metadata entries.")
        
        # Create pairs
        self.instances = list(zip(self.data_list, list(map(self.___get_info, self.metadata_list))))

        # Init counters to support workers
        self.start = 0
        self.end = len(metadata_list)

        
    def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = 0
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)

         return iter(self.instances[iter_start:iter_end])

    def __len__(self):
        return self.end