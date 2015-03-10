#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# This code includes modifications of the Vispy example at:
# https://github.com/vispy/vispy/tree/5098146746c098c7af9e4e09ea44ff6ef5ea2187/examples/demo/gloo/realtime_signals.py
# See the file "LICENSE_vispy.txt" for the copyright license associated with that original example.
#
# Otherwise, modifications made herein are considered copyright under the terms described in the file "LICENSE_actichampy.txt"


from vispy import gloo
from vispy import app
import numpy as np
import math
import OpenGL.GL as gl


VERT_SHADER = """
#version 120

// y coordinate of the position.
attribute float a_position;

// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;

// Size of the table.
uniform float u_nrows;

// Number of samples per signal.
uniform float u_n;

// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;

void main() {

    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x, a_position);

    // Find the affine transformation for the subplots.
    vec2 a = vec2(1., 1./u_nrows)*.9;
    vec2 b = vec2( -1 + 2 * (a_index.x+.5) ,
                   -1 + 2 * (a_index.y+.5) / u_nrows );

    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*position+b, 0.0, 1.0);

    v_index = a_index;

    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120

varying vec3 v_index;

varying vec2 v_position;
varying vec4 v_ab;

void main() {
    gl_FragColor = vec4(1.,1.,1.,0.9);

    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;

    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1) || (test.y > 1))
        discard;
}
"""


class Canvas(app.Canvas):
    def update_index(self):
        self.index = np.c_[np.repeat(np.repeat(0, self.nrows), self.samples_per_screen),
              np.repeat(np.tile(np.arange(self.nrows), 1), self.samples_per_screen),
              np.tile(np.arange(self.samples_per_screen), self.nrows)].astype(np.float32)
    def __init__(self,data):
        self.data = data
        self.samples_per_screen = 10000#self.data.shape[1]
        self.show_latest = True
        self.scroll_time = False
        self.last_sample_to_show = self.data.shape[1]#None
        self.nrows = self.data.shape[0]
        self.update_index() #creates self.index
        app.Canvas.__init__(self, title='Use your wheel to zoom!',keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = self.data.reshape(-1, 1)
        self.program['a_index'] = self.index
        self.program['u_nrows'] = self.nrows
        self.program['u_n'] = self.samples_per_screen
        # self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST )

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, self.width, self.height)
    def on_key_press(self, event):
        if event.key.name=='Space': #toggle show_latest on/off
            self.show_latest = not self.show_latest
            self.last_sample_to_show = self.data.shape[1]
        elif event.key.name=='Z': #toggle scoll_time on
            self.scroll_time = True
        elif 
        else:
            print event.key.name
    def on_key_release(self, event):
        if event.key.name=='Z': #toggle scoll_time back off
            self.scroll_time = False
    def on_mouse_wheel(self, event):
        # dx = np.sign(event.delta[1]) * .05 * self.samples_per_screen
        # self.last_sample_to_show = int(self.last_sample_to_show * math.exp(.1*dx))
        # print dx
        if not self.scroll_time:
            dy = np.sign(event.delta[1]) * .05
            self.samples_per_screen = int(self.samples_per_screen * math.exp(2.5*dy))
            if self.samples_per_screen<100:
                self.samples_per_screen = 100
            elif self.samples_per_screen>self.data.shape[1]:
                self.samples_per_screen = self.data.shape[1]
        elif not self.show_latest:
            self.last_sample_to_show = self.last_sample_to_show + self.samples_per_screen*(event.delta[0]/100)
            if self.last_sample_to_show<self.samples_per_screen:
                self.last_sample_to_show = self.samples_per_screen
            elif self.last_sample_to_show>self.data.shape[1]:
                self.last_sample_to_show = self.data.shape[1]
    # def on_timer(self, event):
    def on_draw(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        k = int(1000/60) #emulating 1kHz new data rate (assuming this function gets called every refresh at 60Hz refresh rate)
        self.data = np.c_[self.data,amplitudes * np.random.randn(self.nrows, k)]
        if self.samples_per_screen > self.data.shape[1]:
            self.samples_per_screen = self.data.shape[1]
            this_data = self.data
        else:
            if self.show_latest:
                this_data = self.data[:,-self.samples_per_screen:]
            else:
                this_data = self.data[:,(self.last_sample_to_show-self.samples_per_screen):self.last_sample_to_show]
        self.update_index()
        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        self.program['a_index'] = self.index
        self.program['u_n'] = self.samples_per_screen
        self.program['a_position'] = this_data.reshape(-1, 1).astype(np.float32)
        self.update()
        gloo.clear()
        self.program.draw('line_strip')

if __name__ == '__main__':
    nrows = 32
    n = 100000
    amplitudes = .1 + .2 * np.random.rand(nrows, 1).astype(np.float32)
    y = amplitudes * np.random.randn(nrows, n).astype(np.float32)
    c = Canvas(data=y)
    c.show()
    app.run()
