

class AsyncLineReader:
    """
        后台线程按行读取一串文件
        每遇到文件结束时放一个None 作为EOF标记
        queue_max_lines 决定预取深度，而不是整个文件
    """
    def __init__(
        self,
        file-list:list[path],
        queue_max_lines:int=512,
        shuffle_each_epoch:bool=True,
        jitter_sec:float=2.0
    ):
        self.files=file_list
        self.queue=Queue(maxsize=queue_max_lines)
        self.shuffle_each_epoch=shuffle_each_epoch
        self.rng=random.Random(int.from_bytes(file_list[0].stem.encode(),byteorder="big"))
        self._thread=Thread(target=self._worker,daemon=True)
        self._thread.start()
        
    def get(self)->tuple[Path | None, str | None]:
        return self.queue.get()
    
    def _worker(self):
        while True:
            files=self.files[:]
            if self.shuffle_each_epoch:
                random.shuffle(files)
            for fp in files:
                sleep_secs=self.rng.uniform(0,sekf.jitter_sec)
                time.sleep(sleep_secs)
                print(f"AsyncLineReader on file {fp.stem} | sleep {sleep_secs:.2f} done")
                with fp.open("r",encoding="utf-8") as f:
                    for line in f:
                        line=line.rstrip("\n")
                        if line:
                            sekf.queue.put((fp.stem,line))
                    self.queue.put((fp.stem,None))


class TCodeIterableDataset(IterableDataset):
    def__init__(
        self,
        data_dir:Path,
        tokenizer:PreTrainedTokenizerFast,
        max_len:int=4096,
        shuffle_files:bool=True,
        queue_max_lines:int =512,
        suffix:str="jsonl"
    ):
        super().__init__()
        self.data_dir=data_dir
        self.tokenizer=tokenizer
        self.max_len=max_len

        self.shuffle_files=shuffle_files
        self.queue_max_lines=queue_max_lines
        
        self.files=sorted(data_dir.glob(f"*.{suffix}"))
    
    def _get_worker_files(self):
        
        worker_info=torch.utils.data.get_worker_info()
        self.logger.info(f"worker_info {worker_info}")
        if worker_info is None: #单进程
            return self.files
        
        rank=dist.get_rank() if dist.is_initialized() else 0
        world_size =dist.get_world_size() if dist.is_initialized() else 1

        #dataloader worker
        w_info=torch.utils.data.get_worker_info()
        num_workers=w_info.num_workers if w_info is not None else 1
        worker_id_local=w_info.id if w_info is not None else 0
        total_workers=num_workers*world_size
        global_worker_id=rank*num_workers+worker_id_local
        
        if len(self.files)>=total_workers:
            worker_files=self.files[global_worker_id::total_workers]
        else:
            tiled=list(islice(cycle(self.files),total_workers))
            worker_files=tiled[global_worker_id]
            
        return worker_files
    
    def __iter__(self):
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.world_size=dist.get_world_size() if dist.is_initialized() else 1
        self.logger=setup_logger(name='eval_data_logger',rank=self.rank)
        
        worker_files=AsyncLineReader(
            worker_files,
            queue_max_lines=self.queue_max_lines,
            shuffle_each_epoch=self.shuffle_files
        )
        
        num_err_lines_of_file=0
        while True:
            _,line = loader.get()
            if line is None:
                num_err_lines_of_file=0
                continue
        
            try:
                obj=json.loads(line)
                trans_list=obj["trans"]
                
                n_chunks=32 if len(trans_list) >= self.max_len // 16 else 1

                for _ in range(n_chunks):
                    if len(trans_list) >= self.max_len // 16:
                        trans_sub =  trans_list[random.randint(0, len(trans_list) - self.max_len // 16):]
                    else:
                        trans_sub = trans_list[:]
                    tokens=[self.tokenizer.bos_token]
                    prev_tts=-1
                    for trans in trans_sub:
                        _cat,prev_tts = deltatime_to_cat(prev_tts,trans[9])
                        tokens.append(f"Fake::{trans[0]}::{trans[1]}::{trans[2]}")
                        tokens.append(f"Time::{trans[4]}::{trans[5]}::{trans[6]}")
                        tokens.append(f"Shoudan::{trans[10]}::{trans[11]}")
                        tokens.append(f"Trx::{trans[12]}::{trans[13]}::{trans[14]}::{trans[15]}")
                        tokens.append(f"Merchant::{trans[16]}::{trans[17]}")
                        tokens.append(f"TDelta::{_cat}")
                        tokens.append(f"Money::{min(math.ceil(math.log2(trans[19]+1)),30)}")
                        tokens.append(trans[18])
                        tokens.append(self.tokenizer.step_token)
                        
                    tokens=self.tokenizer.encode(''.join(tokens),add_special_tokens=False)
                    
                    yield {
                        "input_ids":  tokens[:self.max_len]
                    }

            exceot Exception as e:
                new_err_lines_of_file+=1
                if num_err_lines_of_file>=10
                    while True:
                        _,line=loader.get()
                        if line is None:
                            num_err_lines_of_file=0
                            break