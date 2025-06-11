from typing import List

from vtk import (
    vtkAppendPolyData,
    vtkPolyData,
    vtkXMLPolyDataReader,
    vtkXMLPolyDataWriter,
)


def combine_vtp_files(input_files: List[str], output_file: str) -> None:
    """
    Combine multiple VTP files into a single VTP file.

    Args:
    - input_files (list[str]): List of paths to the input VTP files to be combined.
    - output_file (str): Path to save the combined VTP file.
    """
    reader = vtkXMLPolyDataReader()
    append = vtkAppendPolyData()

    for file in input_files:
        reader.SetFileName(file)
        reader.Update()
        polydata = vtkPolyData()
        polydata.ShallowCopy(reader.GetOutput())
        append.AddInputData(polydata)

    append.Update()

    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(append.GetOutput())
    writer.Write()
