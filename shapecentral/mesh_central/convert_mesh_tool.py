import math
import re


def convert_to_old_format(
    mesh_fname: str, save_fname: str = None, copy_lines: bool = False
):
    if save_fname is None:
        save_fname = mesh_fname
    conv = Converter(mesh_fname, save_fname, copy_lines)
    conv.read()
    conv.replace()
    conv.write()


def change_vtk_version(filename: str, v: float = 5.1):
    with open(filename, "r") as f:
        x = f.read()
    x = re.sub(r"(# vtk DataFile Version) (.+)", f"\\1 {v}", x)
    with open(filename, "w") as f:
        f.write(x)


class Converter:
    def __init__(self, inp, out, copy_lines=False):
        self.inp = inp
        self.out = out
        self.copy_lines = copy_lines  # if line cells should be copied
        self.original = None
        self.lines = None
        self.polys = None
        self.cells = None

    def read(self):
        with open(self.inp, "r") as f:
            self.original = f.read().split("\n")
        lines_original = list(map(lambda x: x.strip().split(), self.original))

        for i, l in enumerate(self.original):
            if "LINES" in l:
                self.lines = NewContent("LINES", lines_original, i)
            elif "POLYGONS" in l:
                self.polys = NewContent("POLYGONS", lines_original, i)
            elif "CELLS" in l:
                self.cells = NewContent("CELLS", lines_original, i)

    def replace(self):
        if self.polys is not None:
            self.original = self.polys.replace(self.original)
        if self.cells is not None:
            self.original = self.cells.replace(self.original)
        if self.lines is not None:
            self.original = self.lines.replace(self.original, replace=self.copy_lines)

    def write(self):
        with open(self.out, "w") as f:
            f.write("\n".join(self.original))

        change_vtk_version(self.out, 4.2)


class NewContent:
    def __init__(self, kw, content, ln):
        self.kw = kw
        self.ln = ln
        self.name = content[ln][0]
        self.no = int(content[ln][1])
        self.nc = int(content[ln][2])

        flat_list = [item for line in content[ln + 2 :] for item in line]
        flat_list = list(filter("".__ne__, flat_list))

        self.offsets = list(map(int, flat_list[0 : self.no]))
        self.connectivity = list(
            map(int, flat_list[self.no + 2 : self.no + 2 + self.nc])
        )

    @property
    def remove(self):
        return self.ln, self.ln + math.ceil(self.no / 9) + math.ceil(self.nc / 9) + 3

    def replace(self, lines, replace=True):
        nb_cells = self.no - 1
        new_content = []
        if replace:
            new_content = [f"{self.kw} {nb_cells} {nb_cells + self.nc}"]
            for i in range(nb_cells):
                nb_points = self.offsets[i + 1] - self.offsets[i]
                ids = self.connectivity[self.offsets[i] : self.offsets[i + 1]]
                new_content.append(f"{nb_points} {' '.join(map(str, ids))}")

        lines_to_keep = lines
        a, b = self.remove
        del lines_to_keep[a:b]
        lines_to_keep[a:a] = new_content
        lines_to_keep = list(filter("".__ne__, lines_to_keep))
        return lines_to_keep
