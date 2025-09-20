import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_boxes(boxes, title="Boxes"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    for (x, y, w, h) in boxes:
        rect = patches.Rectangle((x, y), w, h, 
                                 linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    plt.gca().invert_yaxis()  # flip y-axis to image-like coordinates
    plt.show()

def is_contained(small_box, large_box, tol=0.9):
    # small_box, large_box: (x,y,w,h)
    xi, yi, wi, hi = small_box
    xo, yo, wo, ho = large_box
    xi1 = xi+wi
    yi1 = yi+hi
    xo1 = xo+wo
    yo1 = yo+ho
    inter_x0 = max(xi, xo)
    inter_y0 = max(yi, yo)
    inter_x1 = min(xi1, xo1)
    inter_y1 = min(yi1, yo1)
    inter_area = max(0, inter_x1-inter_x0) * max(0, inter_y1-inter_y0)
    return inter_area >= tol*(wi*hi)

def remove_nested_boxes(boxes, tolerance=0.9):   
    """Remove boxes that are fully or mostly contained within another box."""
    # first sort boxes by area
    boxes = sorted(boxes, key=lambda b: b[2]*b[3])
    keep = []
    # loop through boxes, smallest to largest
    for i, box in enumerate(boxes):
        drop = False
        # Compare smallest box to all larger boxes
        for j in range(i+1, len(boxes)):
            # if box[i] is contained at least in one of the larger boxes, drop it
            if is_contained(boxes[i], boxes[j], tol=tolerance):
                drop = True
                break
        if not drop:
            keep.append(box)
        
    return keep

def main():
    boxes = [(10,10,20,20), (12,12,5,5), (15,15,10,10), (50,50,20,20), (55,55,5,5)]
    print("Original boxes:", boxes)
    filtered_boxes = remove_nested_boxes(boxes, tolerance=0.9)
    print("Filtered boxes:", filtered_boxes)
    draw_boxes(boxes, "Original Boxes")
    draw_boxes(filtered_boxes, "Filtered Boxes")

if __name__ == "__main__":
    main()