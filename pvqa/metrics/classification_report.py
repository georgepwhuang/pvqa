import torch
from torch import nn


def classification_report(data_dict, y_type):
    non_label_keys = ["accuracy", "micro_avg", "macro avg", "weighted avg"]
    digits = 2

    target_names = [
        "%s" % key for key in data_dict.keys() if key not in non_label_keys
    ]

    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary")

    headers = ["precision", "recall", "f1-score", "support"]
    p = [data_dict[l][headers[0]] for l in target_names]
    r = [data_dict[l][headers[1]] for l in target_names]
    f1 = [data_dict[l][headers[2]] for l in target_names]
    s = [data_dict[l][headers[3]] for l in target_names]

    rows = zip(target_names, p, r, f1, s)

    average_options = ("micro", "macro", "weighted")

    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, 20, digits)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)
    report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average == "micro" and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        if line_heading == "accuracy":
            avg = list(data_dict[line_heading].values())
            row_fmt_accuracy = "{:>{width}s} " + \
                               " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + \
                               " {:>9}\n"
            report += row_fmt_accuracy.format(line_heading, "", "",
                                              *avg, width=width,
                                              digits=digits)
        else:
            avg = list(data_dict[line_heading].values())
            report += row_fmt.format(line_heading, *avg,
                                     width=width, digits=digits)
    return report


class ClassificationReport(nn.Module):
    def __init__(self, num_classes, labels, multilabel=False):
        super().__init__()
        self.num_classes = num_classes
        self.labels = labels
        self.multilabel = multilabel
        self.register_buffer('metrics', torch.zeros(
            [num_classes, 5], dtype=torch.int32))

    def update(self, stat_scores: torch.Tensor):
        self.metrics = self.metrics.add(stat_scores)

    def compute(self):
        tp = self.metrics[:, 0]
        fp = self.metrics[:, 1]
        tn = self.metrics[:, 2]
        fn = self.metrics[:, 3]
        support = self.metrics[:, 4]

        support_sum = torch.sum(support)

        if self.multilabel:
            prcn_micro = torch.div(torch.sum(tp), torch.sum(
                tp) + torch.sum(fp)).nan_to_num()
            rcll_micro = torch.div(torch.sum(tp), torch.sum(
                tp) + torch.sum(fn)).nan_to_num()
            fone_micro = torch.div(
                2 * prcn_micro * rcll_micro, prcn_micro + rcll_micro).nan_to_num()
        else:
            accuracy = torch.div(torch.sum(tp), torch.sum(support))

        prcn_categ = torch.div(tp, tp + fp).nan_to_num()
        rcll_categ = torch.div(tp, tp + fn).nan_to_num()
        fone_categ = torch.div(2 * prcn_categ * rcll_categ,
                               prcn_categ + rcll_categ).nan_to_num()

        prcn_macro = torch.mean(prcn_categ)
        rcll_macro = torch.mean(rcll_categ)
        fone_macro = torch.mean(fone_categ)

        prcn_weigh = torch.div(
            torch.sum(prcn_categ * support), support_sum).nan_to_num()
        rcll_weigh = torch.div(
            torch.sum(rcll_categ * support), support_sum).nan_to_num()
        fone_weigh = torch.div(
            torch.sum(fone_categ * support), support_sum).nan_to_num()

        support = support.cpu().tolist()

        if self.multilabel:
            prcn_micro = prcn_micro.cpu().tolist()
            rcll_micro = rcll_micro.cpu().tolist()
            fone_micro = fone_micro.cpu().tolist()
        else:
            accuracy = accuracy.cpu().tolist()

        prcn_categ = prcn_categ.cpu().tolist()
        rcll_categ = rcll_categ.cpu().tolist()
        fone_categ = fone_categ.cpu().tolist()

        prcn_macro = prcn_macro.cpu().tolist()
        rcll_macro = rcll_macro.cpu().tolist()
        fone_macro = fone_macro.cpu().tolist()

        prcn_weigh = prcn_weigh.cpu().tolist()
        rcll_weigh = rcll_weigh.cpu().tolist()
        fone_weigh = fone_weigh.cpu().tolist()

        support_sum = support_sum.cpu().tolist()

        output = {}
        for i, label in enumerate(self.labels):
            output[label] = {"precision": prcn_categ[i],
                             "recall": rcll_categ[i],
                             "f1-score": fone_categ[i],
                             "support": support[i]}
        if self.multilabel:
            output["micro avg"] = {"precision": prcn_micro,
                                   "recall": rcll_micro,
                                   "f1-score": fone_micro,
                                   "support": support_sum}
        else:
            output["accuracy"] = {"f1-score": accuracy,
                                  "support": support_sum}
        output["macro avg"] = {"precision": prcn_macro,
                               "recall": rcll_macro,
                               "f1-score": fone_macro,
                               "support": support_sum}
        output["weighted avg"] = {"precision": prcn_weigh,
                                  "recall": rcll_weigh,
                                  "f1-score": fone_weigh,
                                  "support": support_sum}
        y_type = "multilabel" if self.multilabel else "multiclass"
        return classification_report(output, y_type)

    def reset(self):
        device = self.metrics.device
        self.metrics = torch.zeros(
            [self.num_classes, 5], dtype=torch.int32, device=device)
