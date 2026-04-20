import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import os

# Configure input and output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_CSV = os.path.join(SCRIPT_DIR, 'data', 'timeline_data.csv')
OUTPUT_PNG = os.path.join(PROJECT_ROOT, 'assets', 'timeline_output.png')

# Color configuration
COLORS = {
    'Continuous': '#FCE4D6',  # Light orange
    'Discrete': '#E2F0D9',    # Light green
    'Border_Validated': '#1F3864', # Dark blue border
}

def get_rounded_rect_path(x, y, w, h, r_inch, ax):
    """Generate a rectangular path with perfect rounded corners to avoid border distortion"""
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width_inch = bbox.width
    height_inch = bbox.height
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    # Convert physical inch radius to radius in data coordinate system
    rx = r_inch * (x_range / width_inch)
    ry = r_inch * (y_range / height_inch)
    
    # Bezier curve control point ratio
    kappa = 0.552284749831
    cx = kappa * rx
    cy = kappa * ry
    
    verts = [
        (x + rx, y),
        (x + w - rx, y),
        (x + w - rx + cx, y), (x + w, y + ry - cy), (x + w, y + ry),
        (x + w, y + h - ry),
        (x + w, y + h - ry + cy), (x + w - rx + cx, y + h), (x + w - rx, y + h),
        (x + rx, y + h),
        (x + rx - cx, y + h), (x, y + h - ry + cy), (x, y + h - ry),
        (x, y + ry),
        (x, y + ry - cy), (x + rx - cx, y), (x + rx, y),
        (0, 0)
    ]
    
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.LINETO,
        mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.CLOSEPOLY
    ]
    
    return mpath.Path(verts, codes)

def generate_timeline(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: Input file {csv_path} not found")
        return

    # Read CSV data
    df = pd.read_csv(csv_path)

    # --- Layout Constants ---
    BOX_WIDTH = 0.85
    BOX_HEIGHT = 0.25
    BOX_SPACING = 0.05
    TIMELINE_Y = 0.0
    TIMELINE_TO_BOX_SPACING = 0.10
    YEAR_TEXT_SPACING = 0.15
    
    # Derived layout constants
    VERTICAL_STEP = BOX_HEIGHT + BOX_SPACING
    FIRST_BOX_Y_CENTER = TIMELINE_Y - TIMELINE_TO_BOX_SPACING - (BOX_HEIGHT / 2)
    
    # Legend position (2 boxes below the 4th box of 2022)
    # 2022 has 4 boxes (index 0, 1, 2, 3), so lowest box is at index 3
    LOWEST_2022_BOX_Y = FIRST_BOX_Y_CENTER - 3 * VERTICAL_STEP
    LEGEND_Y_START = LOWEST_2022_BOX_Y - 2 * VERTICAL_STEP
    
    # Calculate bounds
    X_MIN, X_MAX = 0.2, 7.0
    Y_MAX = TIMELINE_Y + YEAR_TEXT_SPACING + 0.25
    Y_MIN = LEGEND_Y_START - 2 * VERTICAL_STEP - (BOX_HEIGHT / 2) - 0.2
    
    # Calculate figsize to maintain equal aspect ratio
    FIGSIZE_X = 15.0
    FIGSIZE_Y = FIGSIZE_X * ((Y_MAX - Y_MIN) / (X_MAX - X_MIN))

    # Create chart
    fig, ax = plt.subplots(figsize=(FIGSIZE_X, FIGSIZE_Y), dpi=300)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('equal')
    ax.axis('off') # Hide axes

    # Draw timeline (main line)
    ax.plot([0.5, 6.7], [TIMELINE_Y, TIMELINE_Y], color='black', linewidth=3, zorder=1)
    # Draw arrow
    ax.arrow(6.7, TIMELINE_Y, 0.05, 0, head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=3, zorder=1)

    # Map years to X coordinates
    years = sorted(df['Year'].unique())
    year_x_map = {year: i + 1 for i, year in enumerate(range(2021, 2027))}

    # Draw year nodes and text
    for year, x in year_x_map.items():
        ax.plot(x, TIMELINE_Y, marker='o', color='black', markersize=8, zorder=2)
        ax.text(x, TIMELINE_Y + YEAR_TEXT_SPACING, str(year), ha='center', va='bottom', fontsize=16, fontweight='bold', fontfamily='serif')

    # Draw model boxes
    corner_radius_inch = 0.05  # Control physical radius of rounded corners (inches)

    for year in years:
        year_data = df[df['Year'] == year]
        x_center = year_x_map[year]
        
        for i, row in enumerate(year_data.itertuples()):
            y_center = FIRST_BOX_Y_CENTER - i * VERTICAL_STEP
            
            # Determine background and border colors
            bg_color = COLORS[row.Type]
            if row.Validated == 'Yes':
                edge_color = COLORS['Border_Validated']
                line_width = 2
            else:
                edge_color = bg_color # No obvious border
                line_width = 0
            
            # Use custom path to draw rounded rectangle to avoid border distortion
            path = get_rounded_rect_path(x_center - BOX_WIDTH/2, y_center - BOX_HEIGHT/2, BOX_WIDTH, BOX_HEIGHT, corner_radius_inch, ax)
            rect = patches.PathPatch(
                path,
                facecolor=bg_color,
                edgecolor=edge_color,
                linewidth=line_width,
                zorder=3
            )
            ax.add_patch(rect)
            
            # Add text
            ax.text(x_center, y_center, row.Model, ha='center', va='center', fontsize=14, fontfamily='serif', zorder=4)

    # Draw legend (bottom left, inserted in the blank space below 2021 and 2022)
    legend_x = 1.0

    # 1. Continuous diffusion
    path1 = get_rounded_rect_path(legend_x - BOX_WIDTH/2, LEGEND_Y_START - BOX_HEIGHT/2, BOX_WIDTH, BOX_HEIGHT, corner_radius_inch, ax)
    rect1 = patches.PathPatch(
        path1,
        facecolor=COLORS['Continuous'],
        edgecolor=COLORS['Continuous'],
        linewidth=0,
        zorder=3
    )
    ax.add_patch(rect1)
    ax.text(legend_x + BOX_WIDTH/2 + 0.2, LEGEND_Y_START, 'Continuous diffusion', ha='left', va='center', fontsize=12, fontweight='bold', fontfamily='serif')

    # 2. Discrete diffusion
    path2 = get_rounded_rect_path(legend_x - BOX_WIDTH/2, LEGEND_Y_START - VERTICAL_STEP - BOX_HEIGHT/2, BOX_WIDTH, BOX_HEIGHT, corner_radius_inch, ax)
    rect2 = patches.PathPatch(
        path2,
        facecolor=COLORS['Discrete'],
        edgecolor=COLORS['Discrete'],
        linewidth=0,
        zorder=3
    )
    ax.add_patch(rect2)
    ax.text(legend_x + BOX_WIDTH/2 + 0.2, LEGEND_Y_START - VERTICAL_STEP, 'Discrete diffusion', ha='left', va='center', fontsize=12, fontweight='bold', fontfamily='serif')

    # 3. Validated under the setup of GPT-2-small
    path3 = get_rounded_rect_path(legend_x - BOX_WIDTH/2, LEGEND_Y_START - 2*VERTICAL_STEP - BOX_HEIGHT/2, BOX_WIDTH, BOX_HEIGHT, corner_radius_inch, ax)
    rect3 = patches.PathPatch(
        path3,
        facecolor='white',
        edgecolor=COLORS['Border_Validated'],
        linewidth=2,
        zorder=3
    )
    ax.add_patch(rect3)
    ax.text(legend_x + BOX_WIDTH/2 + 0.2, LEGEND_Y_START - 2*VERTICAL_STEP, 'Validated under the setup of GPT-2-small', ha='left', va='center', fontsize=12, fontweight='bold', fontfamily='serif')

    # Save image
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0.1)
    print(f"Chart successfully generated and saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_timeline(INPUT_CSV, OUTPUT_PNG)
