import csv
from pathlib import Path


PROJECT_DIR=Path(__file__).resolve().parents[1]
RESULTS_DIR=PROJECT_DIR/"results"
OUT_DIR=Path(__file__).resolve().parent/"out"


def read_history(filename):
    path=RESULTS_DIR/filename
    rows=[]
    with path.open("r", encoding="utf-8") as file:
        reader=csv.DictReader(file, skipinitialspace=True)
        for row in reader:
            rows.append({
                "epoch":int(row["epoch"]),
                "train_loss":float(row["train_loss"]),
                "train_accuracy":float(row["train_accuracy"]),
                "test_loss":float(row["test_loss"]),
                "test_accuracy":float(row["test_accuracy"]),
                "epoch_time":float(row["epoch_time"]),
            })
    return rows


def read_confusion(filename):
    path=RESULTS_DIR/filename
    matrix=[]
    with path.open("r", encoding="utf-8") as file:
        reader=csv.reader(file)
        header=next(reader)
        classes=header[1:]
        for row in reader:
            matrix.append([int(value) for value in row[1:]])
    return classes, matrix


def save_matplotlib_plots(sgd, momentum):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot([row["epoch"] for row in sgd], [row["train_loss"] for row in sgd], label="SGD train")
    plt.plot([row["epoch"] for row in sgd], [row["test_loss"] for row in sgd], label="SGD test")
    plt.plot([row["epoch"] for row in momentum], [row["train_loss"] for row in momentum], label="Momentum train")
    plt.plot([row["epoch"] for row in momentum], [row["test_loss"] for row in momentum], label="Momentum test")
    plt.title("Loss by epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"loss_by_epoch.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot([row["epoch"] for row in sgd], [row["train_accuracy"] for row in sgd], label="SGD train")
    plt.plot([row["epoch"] for row in sgd], [row["test_accuracy"] for row in sgd], label="SGD test")
    plt.plot([row["epoch"] for row in momentum], [row["train_accuracy"] for row in momentum], label="Momentum train")
    plt.plot([row["epoch"] for row in momentum], [row["test_accuracy"] for row in momentum], label="Momentum test")
    plt.title("Accuracy by epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"accuracy_by_epoch.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot([row["epoch"] for row in sgd], [row["epoch_time"] for row in sgd], label="SGD")
    plt.plot([row["epoch"] for row in momentum], [row["epoch_time"] for row in momentum], label="Momentum")
    plt.title("Epoch time")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"epoch_time.png", dpi=160)
    plt.close()

    save_confusion_matplotlib("confusion_sgd.csv", "confusion_sgd_bars.png", "SGD confusion matrix by class")
    save_confusion_matplotlib("confusion_momentum.csv", "confusion_momentum_bars.png", "Momentum confusion matrix by class")


def save_confusion_matplotlib(filename, output_name, title):
    import matplotlib.pyplot as plt

    classes, matrix=read_confusion(filename)
    classes=[chr(ord("A")+i) for i in range(len(classes))]
    correct=[]
    errors=[]
    for i, row in enumerate(matrix):
        row_sum=sum(row)
        correct_value=row[i] if i<len(row) else 0
        correct.append(correct_value)
        errors.append(row_sum-correct_value)

    x=range(len(classes))
    plt.figure(figsize=(10, 5))
    plt.bar(x, correct, label="correct", color="#4f9fd9")
    plt.bar(x, errors, bottom=correct, label="wrong", color="#f08a8a")
    plt.title(title)
    plt.xlabel("True class")
    plt.ylabel("Count")
    plt.xticks(x, classes)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/output_name, dpi=160)
    plt.close()


def svg_start(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def svg_text(x, y, text, size=14, weight="normal", anchor="start"):
    return f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#10263a">{text}</text>'


def save_svg_line_chart(filename, title, series, y_min=None, y_max=None):
    width=900
    height=460
    left=70
    right=30
    top=55
    bottom=60
    chart_w=width-left-right
    chart_h=height-top-bottom
    values=[value for _, points, _ in series for _, value in points]
    epochs=[epoch for _, points, _ in series for epoch, _ in points]
    max_epoch=max(epochs)
    min_epoch=min(epochs)
    y_min=min(values) if y_min is None else y_min
    y_max=max(values) if y_max is None else y_max
    if y_max==y_min:
        y_max=y_min+1.0

    def sx(epoch):
        return left+(epoch-min_epoch)*chart_w/(max_epoch-min_epoch if max_epoch!=min_epoch else 1)

    def sy(value):
        return top+chart_h-(value-y_min)*chart_h/(y_max-y_min)

    svg=svg_start(width, height)
    svg.append(svg_text(width/2, 30, title, 22, "bold", "middle"))
    svg.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#789"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#789"/>')

    for i in range(6):
        y=top+i*chart_h/5
        value=y_max-i*(y_max-y_min)/5
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+chart_w}" y2="{y:.2f}" stroke="#d8e7ef"/>')
        svg.append(svg_text(left-10, y+4, f"{value:.2f}", 12, anchor="end"))

    legend_x=left
    for name, points, color in series:
        path=" ".join([f"{sx(epoch):.2f},{sy(value):.2f}" for epoch, value in points])
        svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<rect x="{legend_x}" y="{height-28}" width="14" height="14" fill="{color}"/>')
        svg.append(svg_text(legend_x+20, height-16, name, 13))
        legend_x+=145

    svg.append(svg_text(width/2, height-8, "Epoch", 13, anchor="middle"))
    svg.append("</svg>")
    (OUT_DIR/filename).write_text("\n".join(svg), encoding="utf-8")


def save_svg_confusion(filename, output_name, title):
    classes, matrix=read_confusion(filename)
    classes=[chr(ord("A")+i) for i in range(len(classes))]
    correct=[]
    errors=[]
    for i, row in enumerate(matrix):
        row_sum=sum(row)
        correct_value=row[i] if i<len(row) else 0
        correct.append(correct_value)
        errors.append(row_sum-correct_value)

    width=900
    height=460
    left=70
    top=55
    bottom=65
    chart_w=780
    chart_h=height-top-bottom
    max_value=max([correct[i]+errors[i] for i in range(len(classes))] or [1])
    bar_gap=14
    bar_w=(chart_w-bar_gap*(len(classes)-1))/len(classes)

    svg=svg_start(width, height)
    svg.append(svg_text(width/2, 30, title, 22, "bold", "middle"))
    svg.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#789"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#789"/>')

    for i, class_name in enumerate(classes):
        x=left+i*(bar_w+bar_gap)
        correct_h=correct[i]*chart_h/max_value
        error_h=errors[i]*chart_h/max_value
        base=top+chart_h
        svg.append(f'<rect x="{x:.2f}" y="{base-correct_h:.2f}" width="{bar_w:.2f}" height="{correct_h:.2f}" fill="#4f9fd9"/>')
        svg.append(f'<rect x="{x:.2f}" y="{base-correct_h-error_h:.2f}" width="{bar_w:.2f}" height="{error_h:.2f}" fill="#f08a8a"/>')
        svg.append(svg_text(x+bar_w/2, height-36, class_name, 13, anchor="middle"))

    svg.append(f'<rect x="{left}" y="{height-24}" width="14" height="14" fill="#4f9fd9"/>')
    svg.append(svg_text(left+20, height-12, "correct", 13))
    svg.append(f'<rect x="{left+110}" y="{height-24}" width="14" height="14" fill="#f08a8a"/>')
    svg.append(svg_text(left+130, height-12, "wrong", 13))
    svg.append("</svg>")
    (OUT_DIR/output_name).write_text("\n".join(svg), encoding="utf-8")


def save_svg_plots(sgd, momentum):
    save_svg_line_chart("loss_by_epoch.svg", "Loss by epoch", [
        ("SGD train", [(row["epoch"], row["train_loss"]) for row in sgd], "#2779bd"),
        ("SGD test", [(row["epoch"], row["test_loss"]) for row in sgd], "#7cc0e8"),
        ("Momentum train", [(row["epoch"], row["train_loss"]) for row in momentum], "#c75c5c"),
        ("Momentum test", [(row["epoch"], row["test_loss"]) for row in momentum], "#f08a8a"),
    ])
    save_svg_line_chart("accuracy_by_epoch.svg", "Accuracy by epoch", [
        ("SGD train", [(row["epoch"], row["train_accuracy"]) for row in sgd], "#2779bd"),
        ("SGD test", [(row["epoch"], row["test_accuracy"]) for row in sgd], "#7cc0e8"),
        ("Momentum train", [(row["epoch"], row["train_accuracy"]) for row in momentum], "#c75c5c"),
        ("Momentum test", [(row["epoch"], row["test_accuracy"]) for row in momentum], "#f08a8a"),
    ], 0.0, 1.0)
    save_svg_line_chart("epoch_time.svg", "Epoch time", [
        ("SGD", [(row["epoch"], row["epoch_time"]) for row in sgd], "#2779bd"),
        ("Momentum", [(row["epoch"], row["epoch_time"]) for row in momentum], "#c75c5c"),
    ])
    save_svg_confusion("confusion_sgd.csv", "confusion_sgd_bars.svg", "SGD confusion matrix by class")
    save_svg_confusion("confusion_momentum.csv", "confusion_momentum_bars.svg", "Momentum confusion matrix by class")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sgd=read_history("history_sgd.csv")
    momentum=read_history("history_momentum.csv")

    try:
        save_matplotlib_plots(sgd, momentum)
        print(f"Saved PNG plots to {OUT_DIR}")
    except ImportError:
        save_svg_plots(sgd, momentum)
        print(f"matplotlib not found, saved SVG plots to {OUT_DIR}")


if __name__=="__main__":
    main()
