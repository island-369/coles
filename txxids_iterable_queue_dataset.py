

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
        max_len:int=4096
    )
    
    