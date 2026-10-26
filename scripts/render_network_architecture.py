"""Draw the recovered v9.8 graph as editable SVG and high-resolution PNG."""
from pathlib import Path
import argparse
import os

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'runs/matplotlib-cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = '#dbeafe'
TAN = '#fff1da'
GREY = '#f1f5f9'
INK = '#172c42'


def canvas(title, size):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.suptitle(title, fontsize=18, fontweight='bold', color=INK, y=.985)
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.94)
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, size=11, dashed=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle='round,pad=0.006,rounding_size=0.012',
                               linewidth=1.15, edgecolor='#537087', facecolor=color,
                               linestyle='--' if dashed else '-'))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, color=INK, linespacing=1.45)


def wire(ax, points, label=None, label_xy=None, dashed=False, color='#4a6075'):
    if len(points) > 2:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                color=color, linewidth=1.25, linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=13, linewidth=1.25, color=color,
                                linestyle='--' if dashed else '-'))
    if label:
        x, y = label_xy or points[len(points)//2]
        ax.text(x, y, label, ha='center', va='center', fontsize=9, color=INK,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})


def save(fig, output, name):
    fig.savefig(output / (name+'.svg'), metadata={'Date': None})
    fig.savefig(output / (name+'.png'), dpi=150)
    plt.close(fig)


def pipeline(output):
    fig, ax = canvas('Recovered JSR v9.8: complete reconstruction boundary', (13, 10))
    box(ax,.23,.92,.40,.085,'RAW frames + ref-to-frame flow\n+ output-grid inverse covariance',GREY,11,True)
    box(ax,.23,.78,.40,.095,'4 x 4 RAW gather\ncount x 14/K, mean, population std',TAN)
    box(ax,.75,.78,.33,.075,'Legacy RGB [B,3,H,W]',TAN)
    box(ax,.23,.63,.40,.105,'144 + RGB3 + structure3 + noise1\nlocal-amplitude canonicalization\nfrozen mean/std affine (302 values)',TAN,10.5)
    box(ax,.23,.465,.40,.105,'Controller: depth-3 U-Net\n151 -> 36 channels\n1,990,660 trainable parameters')
    box(ax,.75,.63,.33,.075,'Same RAW / flow / covariance',GREY,10.5,True)
    box(ax,.75,.465,.38,.105,'Continuous RAW kernel fusion\n12 coefficients per colour\nexp(-z/2 + 5*tanh(q.basis/5))',TAN,10.5)
    box(ax,.50,.30,.76,.095,'Shared radial LCA on learned + legacy (optional)\nRecompute local amplitude from corrected legacy',TAN,11)
    box(ax,.50,.165,.66,.095,'Green-FiLM RefineNet + output factorization\n[learned RGB, legacy RGB]: 6 -> 3 channels\n2,165,607 trainable parameters',BLUE,11)
    box(ax,.50,.060,.49,.055,'High-resolution linear RGB [B,3,H,W]',GREY,11)
    for a,b in [((.23,.876),(.23,.829)),((.23,.729),(.23,.687)),((.23,.574),(.23,.520)),
                ((.438,.465),(.553,.465)),((.75,.588),(.75,.52)),((.75,.412),(.75,.35)),
                ((.50,.247),(.50,.217)),((.50,.112),(.50,.091))]:
        wire(ax,[a,b])
    wire(ax,[(.438,.78),(.579,.78)])
    wire(ax,[(.921,.78),(.98,.78),(.98,.30),(.887,.30)])
    ax.text(.496,.492,'qmap36',fontsize=9,color=INK,ha='center')
    ax.text(.78,.382,'learned RGB',fontsize=9,color=INK,ha='left')
    ax.text(.02,.006,'Blue: trainable network     Sand: explicit operator     Grey: input/output boundary',fontsize=9,color=INK)
    save(fig,output,'pipeline')


def controller(output):
    fig, ax = canvas('Controller: every encoder, skip, decoder and output layer', (14, 10))
    box(ax,.25,.935,.37,.06,'Input [B,151,64,64]',GREY)
    box(ax,.75,.935,.37,.06,'out Conv1x1: 32 -> 36\n1,188 parameters',BLUE,10.5)
    enc=[('enc.0: DoubleConv','151 -> 32 -> 32','[B,32,64,64]','52,768'),
         ('enc.1: DoubleConv','32 -> 64 -> 64','[B,64,32,32]','55,424'),
         ('enc.2: DoubleConv','64 -> 128 -> 128','[B,128,16,16]','221,440')]
    dec=[('dec.2: DoubleConv','concat 64 + 32 = 96 -> 32 -> 32','[B,32,64,64]','36,928'),
         ('dec.1: DoubleConv','concat 128 + 64 = 192 -> 64 -> 64','[B,64,32,32]','147,584'),
         ('dec.0: DoubleConv','concat 256 + 128 = 384 -> 128 -> 128','[B,128,16,16]','590,080')]
    ys=(.77,.57,.37)
    for index,y in enumerate(ys):
        e,d=enc[index],dec[index]
        box(ax,.25,y,.39,.125,f'{e[0]}\n{e[1]}\n{e[2]} | {e[3]} params',BLUE,10.5)
        box(ax,.75,y,.39,.125,f'{d[0]}\n{d[1]}\n{d[2]} | {d[3]} params',BLUE,10)
        wire(ax,[(.451,y),(.549,y)],'skip',(.5,y+.025),True)
    box(ax,.50,.15,.44,.12,'mid: DoubleConv\n128 -> 256 -> 256\n[B,256,8,8] | 885,248 params',BLUE,10.5)
    wire(ax,[(.25,.901),(.25,.839)])
    wire(ax,[(.75,.839),(.75,.901)])
    for top,bottom in ((.77,.57),(.57,.37)):
        wire(ax,[(.25,top-.068),(.25,bottom+.068)],'AvgPool2x2',(.25,(top+bottom)/2))
        wire(ax,[(.75,bottom+.068),(.75,top-.068)],'bilinear resize',(.75,(top+bottom)/2))
    wire(ax,[(.25,.301),(.25,.15),(.272,.15)],'AvgPool2x2',(.25,.24))
    wire(ax,[(.728,.15),(.75,.15),(.75,.301)],'bilinear resize',(.79,.24))
    ax.text(.5,.035,'DoubleConv = Conv3x3 + GELU + Conv3x3 + GELU; bias=True, padding=1\nAverage pooling floors odd sizes; bilinear uses align_corners=False and the skip shape. Total: 1,990,660 parameters.',
            ha='center',va='center',fontsize=10,color=INK,linespacing=1.7)
    save(fig,output,'controller')


def refinenet(output):
    fig, ax = canvas('RefineNet: all eight Green-FiLM residual blocks', (13, 14))
    box(ax,.42,.94,.72,.07,'X = [learned RGB, legacy RGB] / s\n[B,6,H,W]; s = amplitude / 0.34523068818006963',TAN,11)
    box(ax,.35,.825,.43,.07,'inp Conv3x3: 6 -> 116\n6,380 parameters',BLUE,11)
    box(ax,.81,.825,.29,.07,'guide Conv3x3: 2 -> 116\ninput greens X[:,1] and X[:,4]\n2,204 parameters',BLUE,10)
    wire(ax,[(.35,.901),(.35,.866)])
    wire(ax,[(.71,.902),(.81,.902),(.81,.865)])
    ys=[.725-i*.061 for i in range(8)]
    for i,y in enumerate(ys):
        box(ax,.35,y,.48,.043,f'Block {i}: PreActResidual + Green-FiLM  |  269,584 params',BLUE,9.5)
        wire(ax,[(.35,.785 if i==0 else ys[i-1]-.026),(.35,y+.027)])
        wire(ax,[(.81,y),(.596,y)])
        ax.text(.703,y+.012,f'gamma_{i}, beta_{i}',ha='center',fontsize=8,color=INK)
    ax.plot([.81,.81],[.784,ys[-1]],color='#4a6075',linewidth=1.25)
    box(ax,.35,.19,.48,.058,'out Conv1x1: 116 -> 3\n[dG, d(R-G), d(B-G)] | 351 parameters',BLUE,10)
    wire(ax,[(.35,ys[-1]-.027),(.35,.223)])
    box(ax,.42,.077,.73,.105,'Add colour-difference residuals to normalized learned RGB\nclip(normalized RGB, 0, 1) * s\nOutput [B,3,H,W]',TAN,11)
    wire(ax,[(.35,.157),(.35,.132)])
    ax.text(.80,.19,'One guide feature is shared\nby all eight blocks.\nEach block has its own\n1x1 gamma and beta heads.',ha='center',va='center',fontsize=9.5,color=INK,linespacing=1.6)
    ax.text(.50,.002,'All feature maps stay H x W. No BatchNorm, Dropout or hidden attention. Total: 2,165,607 parameters.',ha='center',fontsize=9,color=INK)
    save(fig,output,'refinenet')


def residual_block(output):
    fig, ax = canvas('One RefineNet block: exact pre-activation residual and FiLM order', (16, 7))
    box(ax,.045,.49,.065,.10,'h\n116ch',GREY,10)
    for x,label in ((.17,'GELU'),(.30,'Conv3x3\n116 -> 116'),(.43,'GELU'),(.56,'Conv3x3\n116 -> 116')):
        box(ax,x,.49,.095,.12,label,BLUE,10)
    box(ax,.68,.49,.055,.10,'+',TAN,16)
    box(ax,.80,.49,.055,.10,'x',TAN,16)
    box(ax,.92,.49,.055,.10,'+',TAN,16)
    xs=(.082,.118,.222,.246,.353,.378,.483,.506,.613,.647,.712,.766,.832,.887,.952,.99)
    for left,right in zip(xs[::2],xs[1::2]):
        wire(ax,[(left,.49),(right,.49)])
    ax.text(.99,.42,'h_next',ha='center',fontsize=10,color=INK)
    wire(ax,[(.045,.547),(.045,.69),(.68,.69),(.68,.546)],'identity residual',(.33,.69))
    box(ax,.48,.91,.17,.10,'shared guide\n116ch',GREY,10)
    box(ax,.73,.91,.16,.10,'gamma_i Conv1x1\n116 -> 116',BLUE,10)
    box(ax,.80,.705,.18,.11,'1 + 0.1*tanh(.)',TAN,10.5)
    box(ax,.93,.91,.10,.10,'beta_i Conv1x1\n116 -> 116',BLUE,9)
    wire(ax,[(.572,.91),(.645,.91)])
    wire(ax,[(.73,.853),(.80,.766)])
    wire(ax,[(.80,.644),(.80,.546)])
    wire(ax,[(.48,.966),(.48,.995),(.93,.995),(.93,.966)])
    wire(ax,[(.93,.853),(.93,.63),(.92,.546)])
    ax.text(.50,.15,'u = h + Conv2(GELU(Conv1(GELU(h))))\nh_next = u * (1 + 0.1*tanh(gamma_i(guide))) + beta_i(guide)',
            ha='center',va='center',fontsize=13,color=INK,linespacing=1.8)
    ax.text(.50,.025,'Two 3x3 convolutions: 242,440 parameters; gamma/beta 1x1 heads: 27,144. Block total: 269,584.',ha='center',fontsize=10,color=INK)
    save(fig,output,'refinenet_block')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'jsr-v98-network'})
    for render in (pipeline,controller,refinenet,residual_block):
        render(args.output)
    print('Rendered 4 diagrams as SVG + PNG:',args.output)


if __name__=='__main__':
    main()
