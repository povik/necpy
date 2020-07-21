from vispy import app, gloo, visuals, scene, io
import numpy as np


class DirectionalPlotVisual(visuals.Visual):
    VERTEX = '''
    varying float height;
    void main() {
        vec3 pos = $position;
        height = $height;

        vec4 visual_pos = vec4(pos * height, 1);
        vec4 doc_pos = $visual_to_doc(visual_pos);
        gl_Position = $doc_to_render(doc_pos);
    }
    '''
    FRAGMENT = """
            #version 130
            varying float height;

            float mag2col_base2(float val)
            {
                if (val <= 0.0)
                    return 0.0;
                if (val >= 1.0)
                    return 1.0;

                return val;
            }

            float mag2col_base2_blue(float val)
            {
                if (val <= -2.75)
                    return 0.0;

                if (val <= -1.75)
                    return val + 2.75;

                if (val <= -0.75)
                    return -(val + 0.75);

                if (val <= 0.0)
                    return 0.0;

                if (val >= 1.0)
                    return 1.0;

                return val;
            }

            vec3 mag2col(float a) {
                a *= 2.8;
                a -= 2.3;
                return vec3(mag2col_base2(a + 1.0), mag2col_base2(a),
                            mag2col_base2_blue(a - 1.0));
            }

            void main() {
                //gl_FragColor = vec4(height, 1 - height, height, 1 );
                gl_FragColor = vec4(mag2col(height), 1);
            }
    """

    def __init__(self, resolution):
        visuals.Visual.__init__(self, self.VERTEX, self.FRAGMENT)

        side_size = resolution ** 2 * 4
        self.nvertices = side_size * 6

        mesh = np.zeros((self.nvertices, 3), dtype=np.float32)

        mesh[0:side_size:4,0] = np.zeros(resolution ** 2) + (resolution / 2.0)
        mesh[0:side_size:4,1] = np.tile(np.arange(resolution) - (resolution / 2.0), resolution)
        mesh[0:side_size:4,2] = np.repeat(np.arange(resolution) - (resolution / 2.0), resolution)
        mesh[1:side_size:4,:] = mesh[0:side_size:4] + [0.0, 1.0, 0.0]
        mesh[2:side_size:4,:] = mesh[0:side_size:4] + [0.0, 1.0, 1.0]
        mesh[3:side_size:4,:] = mesh[0:side_size:4] + [0.0, 0.0, 1.0]

        mesh[side_size:side_size * 2] = np.roll(mesh[0:side_size], 1, axis=1)
        mesh[side_size * 2:side_size * 3] = np.roll(mesh[0:side_size], 2, axis=1)

        mesh[side_size * 3:side_size * 6] = -mesh[0:side_size * 3]

        mesh[:] /= np.linalg.norm(mesh, axis=1).reshape(-1, 1)

        vertices = np.concatenate((
            mesh[0::4,:].reshape((-1, 1, 3)),
            mesh[1::4,:].reshape((-1, 1, 3)),
            mesh[2::4,:].reshape((-1, 1, 3)),
            mesh[0::4,:].reshape((-1, 1, 3)),
            mesh[2::4,:].reshape((-1, 1, 3)),
            mesh[3::4,:].reshape((-1, 1, 3)),
        ), axis=1).reshape((-1, 3))
        self.vertices = vertices
        self.shared_program.vert['position'] = gloo.VertexBuffer(vertices)
        self._draw_mode = 'triangles'

        self.set_ampls(lambda v: v[:,0]*0+1)
        self.set_gl_state(depth_test=True)

    def set_ampls(self, f):
        heights = f(self.vertices)
        self.shared_program.vert['height'] = gloo.VertexBuffer(heights)
        self.update()

    def _prepare_transforms(self, view):
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert['visual_to_doc'] = tr.get_transform('visual', 'document')
        view_vert['doc_to_render'] = tr.get_transform('document', 'render')


DirectionalPlot = scene.visuals.create_visual_node(DirectionalPlotVisual)
TextNode = scene.visuals.create_visual_node(visuals.TextVisual)

canvas = scene.SceneCanvas(keys='interactive')
canvas.size = 800, 600
canvas.show()

view = canvas.central_widget.add_view()
vb1 = scene.widgets.ViewBox(border_color='yellow', parent=canvas.scene, pos=(40, 40), size=(260, 260))

vb1.camera, view.camera = scene.TurntableCamera(), scene.TurntableCamera()
vb1.camera.link(view.camera, props=['elevation', 'azimuth'])
vb1.camera.scale_factor = 10.0

mesh = DirectionalPlot(10, parent=view.scene)

scene.visuals.XYZAxis(parent=vb1.scene)
scene.visuals.XYZAxis(parent=view.scene)


from nec import *
v, e = grid(6,1.0,6,1.0)
v = v*np.array([[-1,1,1]])

line = scene.visuals.Line(pos=v[e.flatten()], parent=vb1.scene, width=2, connect='segments')
e_d = np.sqrt(np.sum((v[e[:,1]] - v[e[:,0]])**2, axis=1))
k = 0.3

factors = np.zeros((3, len(e), len(e)))
for j, e_, d in zip(range(len(e)), e, e_d):
    v_m, v_p = e_
    e_m, e_p = np.argwhere((e==v_m) & (e[:,::-1]!=v_p)), np.argwhere((e==v_p) & (e[:,::-1]!=v_m))
    d_m, d_p = e_d[e_m[:,0]], e_d[e_p[:,0]]

    factors_ = reverse(
        base_function(d_m, d, d_p, k=k),
        np.concatenate((e_m[:,1]==0, [False], e_p[:,1]==1))
    )

    factors[:,e_m[:,0],j] = factors_[:,:len(e_m)]
    factors[:,j,j] = factors_[:,len(e_m)]
    factors[:,e_p[:,0],j] = factors_[:,len(e_m)+1:]


def farfield(ii, n):
    ii = ii.reshape((1,-1,3))
    
    n = np.array(n).reshape((-1,1,3))
    e_d = np.sqrt(np.sum((v[e[:,1]] - v[e[:,0]])**2, axis=1)).reshape((1,-1,1))
    s_ = (v[e[:,1]] - v[e[:,0]]).reshape((1,-1,3)) / e_d
    gamma = np.sum(s_*n, axis=2)
    gamma = gamma.reshape(gamma.shape+(1,))

    gamma_ = np.abs(gamma)
    c1 = ii[:,:,0:1]*e_d*np.sinc(k*(e_d/2)*gamma_)
    c2 = ii[:,:,1:2]*e_d*1j/(1+gamma_) * (np.sinc(k*(e_d/2)*(1-gamma_)) - np.sinc(k*e_d/2)*np.cos(k*e_d/2*gamma_)) * np.heaviside(gamma, 0.0)
    c3 = ii[:,:,2:3]*e_d*1/(1+gamma_)  * (np.sinc(k*(e_d/2)*(1-gamma_)) + gamma_*np.sinc(k*e_d/2*gamma_)*np.cos(k*e_d/2))
    
    ph = np.exp(1j*k*np.sum(n*((v[e[:,0]]+v[e[:,1]])/2).reshape(1,-1,3), axis=2))
    ph = ph.reshape(ph.shape + (1,))
    ll = (c1+c2+c3)*(s_-n*gamma)*ph
    return np.sum(ll, axis=1)


def generate():
    rand = np.random.normal(size=(len(e)))+1j*np.random.normal(size=(len(e)))
    ii = np.swapaxes(np.dot(factors, rand), 0, 1)
    print(ii.shape)
    rand_ii = np.swapaxes(np.array((rand, np.zeros_like(rand), np.zeros_like(rand))), 0, 1)
    mesh.set_ampls(lambda v: np.log10(np.linalg.norm(farfield(ii, v), axis=1).astype(np.float32))+1.0)


@canvas.connect
def on_key_press(ev):
    if ev.key == 'Space':
        generate()


#timer = app.Timer()
#@timer.connect
#def update(ev):
#    line.set_phase(ev.elapsed*10)
#timer.start(0)


if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()
