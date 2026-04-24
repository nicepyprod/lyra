# Open Source License Attribution

   Cosmos uses Open Source components. You can find the details of these open-source projects along with license information below, sorted alphabetically.
   We are grateful to the developers for their contributions to open source and acknowledge these below.

<!-- Personal note: forked from nv-tlabs/lyra for learning purposes. Keeping attributions up to date as I add/remove dependencies. -->

## Better-Profanity - [MIT License](https://github.com/snguyenthanh/better_profanity/blob/master/LICENSE)

   ```

   Copyright (c) 2018 The Python Packaging Authority

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## FFmpeg - [FFMPEG License](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)

   ```
   # License

   Most files in FFmpeg are under the GNU Lesser General Public License version 2.1
   or later (LGPL v2.1+). Read the file `COPYING.LGPLv2.1` for details. Some other
   files have MIT/X11/BSD-style licenses. In combination the LGPL v2.1+ applies to
   FFmpeg.

   Some optional parts of FFmpeg are licensed under the GNU General Public License
   version 2 or later (GPL v2+). See the file `COPYING.GPLv2` for details. None of
   these parts are used by default, you have to explicitly pass `--enable-gpl` to
   configure to activate them. In this case, FFmpeg's license changes to GPL v2+.

   Specifically, the GPL parts of FFmpeg are:

   - libpostproc
   - optional x86 optimization in the files
       - `libavcodec/x86/flac_dsp_gpl.asm`
       - `libavcodec/x86/idct_mmx.c`
       - `libavfilter/x86/vf_removegrain.asm`
   - the following building and testing tools
       - `compat/solaris/make_sunver.pl`
       - `doc/t2h.pm`
       - `doc/texi2pod.pl`
       - `libswresample/tests/swresample.c`
       - `tests/checkasm/*`
       - `tests/tiny_ssim.c`
   - the following filters in libavfilter:
       - `signature_lookup.c`
       - `vf_blackframe.c`
       - `vf_boxblur.c`
       - `vf_colormatrix.c`
       - `vf_cover_rect.c`
       - `vf_cropdetect.c`
       - `vf_delogo.c`
       - `vf_eq.c`
       - `vf_find_rect.c`
       - `vf
