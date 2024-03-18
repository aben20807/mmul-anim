#!/usr/bin/env python3
#
# Copyright 2019 Michal Sojka <michal.sojka@cvut.cz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Copyright 2024 Po-Hsuan Huang <aben20807@gmail.com>
# Modifications:
# * reformat by black
# * add main function and avoid global variables
# * more configurable arguments
# * no visualization mode, calculate the hit rate only
# * gif mode (TODO)
# * boundary check (TODO)
# * SIMD (TODO)

import cairo
import argparse
from subprocess import Popen, PIPE
import sys
from enum import Enum
from typing import override
from abc import ABC, abstractmethod


class FileOutputType(Enum):
    pdf = "pdf"
    mp4 = "mp4"

    def __str__(self):
        return self.name


def get_args():
    parser = argparse.ArgumentParser(
        description="Process some integers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Matrix settings
    parser.add_argument(
        "--matrix-size",
        metavar="size",
        type=int,
        default=16,
        help="n of the n-by-n matrix",
    )
    parser.add_argument("--transpose", action="store_true", help="Transpose matrix B")

    # Cache settings
    parser.add_argument(
        "--L1",
        metavar="size",
        type=int,
        default=4,
        help="number of cache lines in L1 cache",
    )
    parser.add_argument(
        "--L2",
        metavar="size",
        type=int,
        default=16,
        help="number of cache lines in L2 cache",
    )
    parser.add_argument(
        "--cache-line",
        metavar="size",
        type=int,
        default=4,
        help="number of elements for each cache line, must be power of 2",
    )

    # Blocking
    parser.add_argument(
        "--block1", metavar="size", type=int, default=16, help="Inner block size"
    )
    parser.add_argument(
        "--block2", metavar="size", type=int, default=4, help="Outer block size"
    )

    # Visualization
    parser.add_argument("--viz", action="store_true", help="To generate pdf or mp4")
    parser.add_argument("--no-memory", action="store_true", help="Do not draw memory")

    # Output settings
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--subtitle", type=str, default="")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="matrix_mul.pdf",
        help="Output PDF file",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=FileOutputType,
        choices=list(FileOutputType),
        default=FileOutputType.pdf,
        help="The type of the output file",
    )
    return parser.parse_args()


def check_output_extension(output: str, type: FileOutputType):
    if type == FileOutputType.pdf and output.lower().endswith("pdf"):
        return
    if type != FileOutputType.pdf and not output.lower().endswith("pdf"):
        return
    print("Output extension mismatch")
    sys.exit(1)


def config(args):
    bigctx = {}
    bigctx["ctx"] = None
    bigctx["surface"] = None
    bigctx["ffmpeg"] = None
    bigctx["args"] = args

    if not args.viz:
        return bigctx
    check_output_extension(args.output, args.type)
    if args.type == FileOutputType.pdf:
        try:
            surface = cairo.PDFSurface(args.output, 380, 200)
        except Exception as e:
            print(f"Error: '{e}', you may need to close the pdf from your viewer.")
            sys.exit(1)
    else:
        png_scale = 3
        surface = cairo.ImageSurface(
            cairo.FORMAT_RGB24, 380 * png_scale, 200 * png_scale
        )
    bigctx["ctx"] = cairo.Context(surface)

    ffmpeg = None
    if args.type != FileOutputType.pdf:
        bigctx["ctx"].scale(png_scale, png_scale)
        ffmpeg = Popen(
            "ffmpeg -y -f png_pipe -r 30 -i - -vcodec h264 -r 30 -f mp4".split()
            + [args.output],
            stdin=PIPE,
        )
    bigctx["ctx"].set_operator(cairo.OPERATOR_SOURCE)
    bigctx["surface"] = surface
    bigctx["ffmpeg"] = ffmpeg
    return bigctx


class Save:
    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        self.ctx.save()

    def __exit__(self, exc_type, exc_value, traceback):
        self.ctx.restore()


class Translate:
    def __init__(self, ctx, dx, dy):
        self.ctx = ctx
        self.dx = dx
        self.dy = dy

    def __enter__(self):
        self.ctx.save()
        self.ctx.translate(self.dx, self.dy)

    def __exit__(self, exc_type, exc_value, traceback):
        self.ctx.restore()


class Scale:
    def __init__(self, ctx, sx, sy):
        self.ctx = ctx
        self.sx = sx
        self.sy = sy

    def __enter__(self):
        self.ctx.save()
        self.ctx.scale(self.sx, self.sy)

    def __exit__(self, exc_type, exc_value, traceback):
        self.ctx.restore()


class Matrix:
    size = None
    L1_size = None
    L2_size = None
    cache_line_size = None

    def __init__(self, name, transpose=False):
        self.cache = list()
        self.last_access = (None, None)
        self.name = name
        self.transpose = transpose
        self.accesses = 0
        self.L1_hits = 0
        self.L2_hits = 0

    @classmethod
    def static_init(cls, args):
        cls.size = args.matrix_size
        cls.L1_size = args.L1
        cls.L2_size = args.L2
        cls.cache_line_size = args.cache_line

    def xy2cache(self, x, y):
        return (y * self.size + x) & ~(self.cache_line_size - 1)

    def cache2xy(self, tag):
        l = list()
        for i in range(self.cache_line_size):
            addr = tag + i
            x, y = (addr % self.size, addr // self.size)
            l.append((x, y))
        return l

    def access(self, y, x):
        if self.transpose:
            x, y = y, x
        self.accesses += 1
        self.last_access = (x, y)
        tag = self.xy2cache(x, y)
        try:
            i = self.cache.index(tag)
            del self.cache[i]
            if i < self.L1_size:
                self.L1_hits += 1
            else:
                self.L2_hits += 1
        except ValueError:
            pass
        self.cache.insert(0, tag)
        if len(self.cache) > self.L2_size:
            del self.cache[self.L2_size]


class MatrixDrawer(ABC):
    ctx = None

    def __init__(self, matrix):
        self.matrix: Matrix = matrix
        self.draw()

    @classmethod
    def static_init(cls, ctx):
        cls.ctx = ctx

    def draw(self):
        ctx = self.ctx
        with Save(ctx):
            self.set_scale()
            self.show_name()
            with Save(ctx):
                if self.matrix.L1_size > 0:
                    self.show_stat(
                        "mem: %-3d L1 hit: %-3d L2 hit: %-3d"
                        % (
                            self.matrix.accesses,
                            self.matrix.L1_hits,
                            self.matrix.L2_hits,
                        )
                    )
                else:
                    self.show_stat(
                        "mem: %-3d cache hit: %-3d"
                        % (self.matrix.accesses, self.matrix.L2_hits)
                    )
            self.draw_cache()
            self.draw_grid()

    @abstractmethod
    def draw_grid():
        raise NotImplementedError

    def draw_cache(self):
        # SIMD (TODO)
        ctx = self.ctx
        for i in range(len(self.matrix.cache)):
            tag = self.matrix.cache[i]
            for x, y in [self.matrix.cache2xy(tag)[0]]:
                with Save(ctx):
                    self.cache_path(x, y)
                    if i < self.matrix.L1_size:
                        t = i * 1 / self.matrix.L1_size / 1.0
                        ctx.set_source_rgb(t, 1, t)
                    else:
                        t = i * 1 / self.matrix.L2_size / 1.5
                        ctx.set_source_rgb(1, t, t)
                    ctx.fill()
        if self.matrix.last_access != (None, None):
            with Save(ctx):
                (x, y) = self.matrix.last_access
                self.element_path(x, y)
                ctx.set_line_width(4 / 10)
                ctx.stroke()


class MatrixDrawerRect(MatrixDrawer):
    def show_name(self):
        ctx = self.ctx
        ctx.set_font_size(Matrix.size / 12)
        ctx.move_to(0, -0.3)
        ctx.show_text(self.matrix.name + "  ")

    def show_stat(self, stat):
        ctx = self.ctx
        ctx.set_font_size(Matrix.size / 20)
        ctx.show_text(stat)

    @override
    def draw_grid(self):
        ctx = self.ctx
        s = self.matrix.size
        for i in range(s):
            ctx.move_to(0, i)
            ctx.line_to(s, i)
            ctx.stroke()
            ctx.move_to(i, 0)
            ctx.line_to(i, s)
            ctx.stroke()

        ctx.move_to(0, 0)
        ctx.line_to(s, 0)
        ctx.line_to(s, s)
        ctx.line_to(0, s)
        ctx.close_path()
        ctx.stroke()

    def set_scale(self):
        ctx = self.ctx
        ctx.scale(1 / self.matrix.size, 1 / self.matrix.size)
        ctx.set_line_width(1 / 10)

    def element_path(self, x, y):
        ctx = self.ctx
        ctx.move_to(x, y)
        ctx.line_to(x + 1, y + 0)
        ctx.line_to(x + 1, y + 1)
        ctx.line_to(x + 0, y + 1)
        ctx.close_path()

    def cache_path(self, x, y):
        ctx = self.ctx
        ctx.move_to(x, y)
        ctx.line_to(x + self.matrix.cache_line_size, y + 0)
        ctx.line_to(x + self.matrix.cache_line_size, y + 1)
        ctx.line_to(x + 0, y + 1)
        ctx.close_path()


class MatrixDrawerLine(MatrixDrawer):
    def show_name(self):
        ctx = self.ctx
        ctx.set_font_size(1.5)
        ctx.move_to(-1.5, 1)
        ctx.show_text(self.matrix.name + "  ")

    @staticmethod
    def show_stat(stat):
        pass

    @override
    def draw_grid(self):
        ctx = self.ctx
        s = self.matrix.size * self.matrix.size
        ctx.move_to(0, 0)
        ctx.line_to(s, 0)
        ctx.line_to(s, 1)
        ctx.line_to(0, 1)
        ctx.close_path()
        ctx.stroke()

    def set_scale(self):
        ctx = self.ctx
        ctx.scale(
            2 / self.matrix.size / self.matrix.size,
            2 / self.matrix.size / self.matrix.size,
        )
        ctx.set_line_width(1 / 10)

    def element_path(self, x, y):
        ctx = self.ctx
        ctx.move_to(y * self.matrix.size + x, 0)
        ctx.rel_line_to(1, 0)
        ctx.rel_line_to(0, 1)
        ctx.rel_line_to(-1, 0)
        ctx.close_path()

    def cache_path(self, x, y):
        ctx = self.ctx
        ctx.move_to(y * self.matrix.size + x, 0)
        ctx.rel_line_to(Matrix.cache_line_size, 0)
        ctx.rel_line_to(0, 1)
        ctx.rel_line_to(-Matrix.cache_line_size, 0)
        ctx.close_path()


class Stats:
    def __init__(self, a, b, c, hasL1):
        ac = a.accesses + b.accesses + c.accesses
        L1 = a.L1_hits + b.L1_hits + c.L1_hits
        L2 = a.L2_hits + b.L2_hits + c.L2_hits
        self.mem = ac
        self.L1h = L1
        self.L2h = L2
        self.L1p = 100 * L1 // ac
        self.L2p = 100 * L2 // ac
        self.cache = L1 + L2
        self.cachep = 100 * self.cache // ac
        self.hasL1 = hasL1

    def __str__(self):
        if self.hasL1:
            return (
                "mem: %(mem)-4d    L1 hits: %(L1h)-4d (%(L1p)2d%%)    L2 hits: %(L2h)-4d (%(L2p)2d%%)    cache hits: %(cache)-4d (%(cachep)2d%%)"
                % self.__dict__
            )
        else:
            return (
                "mem: %(mem)-4d    cache hits: %(cache)-4d (%(cachep)2d%%)"
                % self.__dict__
            )


class FrameDrawer:
    def __init__(self, bigctx, title="", subtitle=""):
        self.ctx = bigctx["ctx"]
        self.args = bigctx["args"]
        self.ffmpeg = bigctx["ffmpeg"]
        self.surface = bigctx["surface"]

        self.title = "Matrix multiplication: " + (
            title if self.args.title == "" else self.args.title
        )
        self.subtitle = subtitle if self.args.subtitle == "" else self.args.subtitle
        if Matrix.L1_size > 0:
            self.subtitle = (
                f"cache line: {Matrix.cache_line_size} elements, "
                + f"L1: {Matrix.L1_size} lines, L2: {Matrix.L2_size} lines"
                + self.subtitle
            )
        else:
            self.subtitle = (
                f"cache line: {Matrix.cache_line_size} elements, "
                + f"cache: {Matrix.L2_size} lines"
                + self.subtitle
            )
        print(self.title)
        print(self.subtitle)

    def _draw_matrices(self, a, b, c):
        ctx = self.ctx
        args = self.args
        with Save(ctx):
            ctx.set_source_rgb(0, 0, 0)
            dist = 1.2
            ctx.translate(20, 25)
            ctx.set_font_size(10)
            ctx.show_text(self.title)
            ctx.stroke()
            ctx.translate(0, 10)
            ctx.set_font_size(6)
            ctx.show_text(self.subtitle)
            ctx.stroke()
            ctx.translate(0, 15)
            ctx.scale(100, 100)
            ctx.set_font_size(1 / 12)
            with Save(ctx):
                MatrixDrawerRect(a)
            with Translate(ctx, 1.05, 0.5):
                ctx.show_text("×")
            with Translate(ctx, dist, 0):
                MatrixDrawerRect(b)
            with Translate(ctx, 2.25, 0.5):
                ctx.show_text("=")
            with Translate(ctx, 2 * dist, 0):
                MatrixDrawerRect(c)
            with Translate(ctx, 0.0, 1.15):
                stat = Stats(a, b, c, args.L1 > 0)
                ctx.show_text(str(stat))

    def _draw_memory(self, a, b, c):
        ctx = self.ctx
        with Save(ctx):
            ctx.set_source_rgb(0, 0, 0)
            dist = 5 / Matrix.size / Matrix.size
            ctx.translate(20, 175)
            ctx.scale(170, 170)
            with Translate(ctx, 0, 0):
                MatrixDrawerLine(a)
            with Translate(ctx, 0, dist):
                MatrixDrawerLine(b)
            with Translate(ctx, 0, 2 * dist):
                MatrixDrawerLine(c)

    def draw_frame(self, a, b, c, frame_cnt):
        args = self.args
        if not args.viz:
            return
        ctx = self.ctx
        ffmpeg = self.ffmpeg
        surface = self.surface
        if True:
            if args.type != FileOutputType.pdf:
                ctx.set_source_rgb(1, 1, 1)
                ctx.paint()

            self._draw_matrices(a, b, c)
            if not args.no_memory:
                self._draw_memory(a, b, c)
            with Save(ctx):
                ctx.translate(362, 192)
                ctx.set_font_size(4)
                ctx.show_text(f"{frame_cnt+1:6d}")
            ctx.stroke()

            if args.type != FileOutputType.pdf:
                surface.write_to_png(ffmpeg.stdin)
            else:
                surface.show_page()


def perform_matrix_multiply(bigctx, a: Matrix, b: Matrix, c: Matrix):
    args = bigctx["args"]

    drawer = FrameDrawer(
        bigctx,
        title=sys._getframe().f_code.co_name,
        subtitle=", "
        + f"blocking: (inner {args.block1}, outer {args.block2}), "
        + f"B {'w/' if args.transpose else 'w/o'} transpose",
    )
    frame_cnt = 0
    block2_size = args.block2
    block1_size = args.block1
    for i2 in range(0, Matrix.size, block2_size):
        for j2 in range(0, Matrix.size, block2_size):
            for k2 in range(0, Matrix.size, block2_size):
                for i1 in range(i2, i2 + block2_size, block1_size):
                    for j1 in range(j2, j2 + block2_size, block1_size):
                        for k1 in range(k2, k2 + block2_size, block1_size):
                            for i in range(i1, i1 + block1_size):
                                for j in range(j1, j1 + block1_size):
                                    for k in range(k1, k1 + block1_size):
                                        c.access(i, j)
                                        a.access(i, k)
                                        b.access(k, j)

                                        # if frame_cnt < 100:
                                        drawer.draw_frame(a, b, c, frame_cnt)
                                        frame_cnt += 1


def perform_matrix_multiply_v0(bigctx, a: Matrix, b: Matrix, c: Matrix):
    drawer = FrameDrawer(
        bigctx,
        title=sys._getframe().f_code.co_name,
        subtitle="",
    )
    frame_cnt = 0
    for i in range(0, Matrix.size):
        for j in range(0, Matrix.size):
            for k in range(0, Matrix.size):
                c.access(i, j)
                a.access(i, k)
                b.access(k, j)

                drawer.draw_frame(a, b, c, frame_cnt)
                frame_cnt += 1


def perform_matrix_multiply_v1(bigctx, a: Matrix, b: Matrix, c: Matrix):
    drawer = FrameDrawer(
        bigctx,
        title=sys._getframe().f_code.co_name,
        subtitle="",
    )
    frame_cnt = 0
    for i in range(0, Matrix.size):
        for k in range(0, Matrix.size):
            for j in range(0, Matrix.size):
                c.access(i, j)
                a.access(i, k)
                b.access(k, j)

                drawer.draw_frame(a, b, c, frame_cnt)
                frame_cnt += 1


def perform_matrix_multiply_v1(bigctx, a: Matrix, b: Matrix, c: Matrix):
    drawer = FrameDrawer(
        bigctx,
        title=sys._getframe().f_code.co_name,
        subtitle="",
    )
    frame_cnt = 0
    for i in range(0, Matrix.size):
        for k in range(0, Matrix.size):
            for j in range(0, Matrix.size):
                c.access(i, j)
                a.access(i, k)
                b.access(k, j)

                drawer.draw_frame(a, b, c, frame_cnt)
                frame_cnt += 1


def perform_matrix_multiply_v2(bigctx, a: Matrix, b: Matrix, c: Matrix):
    block1_size = 16
    block2_size = 4
    drawer = FrameDrawer(
        bigctx,
        title=sys._getframe().f_code.co_name,
        subtitle=", " + f"blocking: (inner {block1_size}, outer {block2_size})",
    )
    frame_cnt = 0
    for i in range(0, Matrix.size, block1_size):
        for j in range(0, Matrix.size, block2_size):
            for ii in range(i, i + block1_size):
                for jj in range(j, j + block2_size):
                    for k in range(0, Matrix.size):
                        c.access(ii, jj)
                        a.access(ii, k)
                        b.access(k, jj)

                        drawer.draw_frame(a, b, c, frame_cnt)
                        frame_cnt += 1


def main():
    args = get_args()
    bigctx = config(args)

    Matrix.static_init(args)
    a = Matrix("A")
    b = Matrix("B", args.transpose)
    c = Matrix("C")

    MatrixDrawer.static_init(bigctx["ctx"])
    perform_matrix_multiply(bigctx, a, b, c)
    # perform_matrix_multiply_v0(bigctx, a, b, c)
    # perform_matrix_multiply_v1(bigctx, a, b, c)
    # perform_matrix_multiply_v2(bigctx, a, b, c)

    print(Stats(a, b, c, args.L1 > 0))
    if bigctx["ffmpeg"] is not None:
        bigctx["ffmpeg"].stdin.close()
        bigctx["ffmpeg"].wait()


if __name__ == "__main__":
    main()
