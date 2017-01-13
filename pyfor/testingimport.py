import laspy

def read_las(las_file):
    return laspy.file.File(las_file, mode = "r")

point_records = read_las(r"C:\Paco\0_25\000001.S.CLAS.las").points
