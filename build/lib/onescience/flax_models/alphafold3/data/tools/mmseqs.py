

"""Library to run Mmseqs from Python."""

import os
import tempfile
import time
import threading

from absl import logging
from onescience.flax_models.alphafold3.data import parsers
from onescience.flax_models.alphafold3.data.tools import msa_tool
from onescience.flax_models.alphafold3.data.tools import subprocess_utils
import shlex


class Mmseqs(msa_tool.MsaTool):
  """Python wrapper of the Mmseqs binary."""

  def __init__(
      self,
      *,
      binary_path: str,
      database_path: str,
      n_cpu: int = 8,
      e_value: float | None = 1e-4,
      max_sequences: int = 5000,
      use_gpu: int=0,
      msa_format_mode: int=4,
      mmseqs_options: str,
      result2msa_options: str,
  ):
    """Initializes the Python Mmseqs wrapper.

    Args:

    """
    self.binary_path = binary_path
    self.database_path = database_path

    subprocess_utils.check_binary_exists(
        path=self.binary_path, name='MMseqs'
    )
    
    if not os.path.exists(self.database_path):
      raise ValueError(f'Could not find Mmseqs database {database_path}')
    
    self.n_cpu = n_cpu
    self.e_value = e_value
    self.max_sequences = max_sequences
    self.use_gpu = use_gpu
    self.msa_format_mode = msa_format_mode
    self.mmseqs_options = mmseqs_options
    self.result2msa_options = result2msa_options

  def query(self, target_sequence: str) -> msa_tool.MsaToolResult:
    """Queries the database using MMseqs."""
    logging.info('Query sequence: %s', target_sequence)
    with tempfile.TemporaryDirectory() as query_tmp_dir:
      input_fasta_path = os.path.join(query_tmp_dir, 'query.fasta')
      subprocess_utils.create_query_fasta_file(
          sequence=target_sequence, path=input_fasta_path
      )

      logging.info('Searching %s', self.database_path)
      search_start_time = time.time()
      
      sequence_db_path = os.path.join(query_tmp_dir, 'queryDB')
      self._run_createdb_command(input_fasta_path, sequence_db_path)

      result_db_path = os.path.join(query_tmp_dir, 'resultDB')
      self._run_search_command(sequence_db_path, result_db_path)

      output_sto_path = os.path.join(query_tmp_dir, 'output.sto')
      self._run_result2msa_command(sequence_db_path, result_db_path, output_sto_path)

      with open(output_sto_path) as f:
        try:
            a3m = parsers.convert_stockholm_to_a3m(
                f,
                max_sequences=self.max_sequences
            )
        except Exception as e: 
            print(f"convert_stockholm_to_a3m failed with error: {e}. Trying convert_mmseqs_stockholm_to_a3m now.")
            f.seek(0)
            try:
                a3m = parsers.convert_mmseqs_stockholm_to_a3m(
                    f,
                    max_sequences=self.max_sequences
                )
            except Exception as e:  
                print(f"convert_mmseqs_stockholm_to_a3m failed with error: {e}. ")

    logging.info(
    'Searching took %.3f seconds from %s',
    time.time() - search_start_time,
    self.database_path,
    )

    return msa_tool.MsaToolResult(
        target_sequence=target_sequence, a3m=a3m, e_value=self.e_value
    )
    
  def _run_createdb_command(self, input_fasta_path: str, output_db_path: str):
        """Runs the MMseqs2 `createdb` command."""
        cmd = [
            self.binary_path,
            'createdb',
            input_fasta_path,
            output_db_path,
        ]
        subprocess_utils.run(
            cmd=cmd,
            cmd_name='MMseqs2 createdb',
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )
      
  _mmseqs_gpu_lock = threading.Lock()
    
  def _run_search_command(self, input_db_path: str, output_db_path: str):
        """Runs the MMseqs2 `search` command."""
        with self._mmseqs_gpu_lock: 
            with tempfile.TemporaryDirectory() as tmp_dir:
                cmd = [
                    self.binary_path,
                    'search',
                    input_db_path,
                    self.database_path,
                    output_db_path,
                    tmp_dir,  # 使用临时目录
                    '--threads',str(self.n_cpu),
                    '-e',str(self.e_value),
                    '--gpu',str(self.use_gpu),
                ]
                cmd.extend(shlex.split(self.mmseqs_options))
                cmd.extend(['--prefilter-mode', '1'])
                subprocess_utils.run(
                    cmd=cmd,
                    cmd_name='MMseqs2 search',
                    log_stdout=False,
                    log_stderr=True,
                    log_on_process_error=True,
                )

  def _run_result2msa_command(self, input_db_path: str, result_db_path: str, output_msa_path: str):
        """Runs the MMseqs2 `result2msa` command."""
        cmd = [
            self.binary_path,
            'result2msa',
            input_db_path,
            self.database_path,
            result_db_path,
            output_msa_path,
            '--msa-format-mode',
            str(self.msa_format_mode),
        ]
        cmd.extend(shlex.split(self.result2msa_options))
        subprocess_utils.run(
            cmd=cmd,
            cmd_name='MMseqs2 result2msa',
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

