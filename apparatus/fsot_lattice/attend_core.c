/* Same attend as consensus.py and lattice.consensus_quantity.

   Operation order matches the Python loops. Build without fast-math and
   without fused multiply-add so the doubles stay on that path.
*/

#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static int collapse_code(double value, double threshold) {
    if (value > threshold) {
        return 2;
    }
    if (value < -threshold) {
        return 0;
    }
    return 1;
}

static double trit_similarity(
    const double *a,
    const double *b_carrier,
    double b_scale,
    int width,
    double threshold
) {
    int acc = 0;
    int i;
    if (width <= 0) {
        return 0.0;
    }
    for (i = 0; i < width; i++) {
        int ta = collapse_code(a[i], threshold);
        int tb = collapse_code(b_scale * b_carrier[i], threshold);
        if (ta == 1 || tb == 1) {
            continue;
        }
        acc += (ta == tb) ? 1 : -1;
    }
    return (double)acc / (double)width;
}

static double position_coherence(const double *carrier, double scale, int width, double threshold) {
    int hot = 0;
    int i;
    if (width <= 0) {
        return 0.0;
    }
    for (i = 0; i < width; i++) {
        if (fabs(scale * carrier[i]) > threshold) {
            hot += 1;
        }
    }
    return (double)hot / (double)width;
}

/* Attend query over one scaled carrier. Returns the axis quantity, or 0
   when the key drops. Writes nothing when dropped is set and the key is out. */
static int key_contributes(
    const double *query,
    const double *carrier,
    double scale,
    int width,
    double collapse,
    double gate,
    double *weight_out
) {
    double weight;
    if (position_coherence(carrier, scale, width, collapse) <= gate) {
        return 0;
    }
    weight = trit_similarity(query, carrier, scale, width, collapse);
    if (weight == 0.0) {
        return 0;
    }
    *weight_out = weight;
    return 1;
}

static double axis_quantity(const double *state, int active, const double *axis, int width) {
    double denom = 0.0;
    double scale = 0.0;
    int i;
    if (active == 0) {
        return 0.0;
    }
    for (i = 0; i < width; i++) {
        denom += axis[i] * axis[i];
    }
    if (denom == 0.0) {
        return 0.0;
    }
    for (i = 0; i < width; i++) {
        scale += state[i] * axis[i];
    }
    return (scale / denom) * (double)active;
}

EXPORT double fsot_consensus_quantity(
    double left,
    double right,
    int sign,
    const double *plus,
    const double *minus,
    int width,
    double collapse,
    double gate
) {
    double plus_state[64];
    double minus_state[64];
    double weight;
    int plus_active = 0;
    int minus_active = 0;
    int i;
    if (width <= 0 || width > 64 || plus == NULL || minus == NULL) {
        return 0.0;
    }
    for (i = 0; i < width; i++) {
        plus_state[i] = 0.0;
        minus_state[i] = 0.0;
    }
    if (key_contributes(plus, plus, left, width, collapse, gate, &weight)) {
        plus_active += 1;
        for (i = 0; i < width; i++) {
            plus_state[i] += weight * (left * plus[i]);
        }
    }
    if (sign > 0 && key_contributes(plus, plus, right, width, collapse, gate, &weight)) {
        plus_active += 1;
        for (i = 0; i < width; i++) {
            plus_state[i] += weight * (right * plus[i]);
        }
    }
    if (plus_active > 0) {
        for (i = 0; i < width; i++) {
            plus_state[i] /= (double)plus_active;
        }
    }
    if (sign < 0 && key_contributes(minus, minus, right, width, collapse, gate, &weight)) {
        minus_active += 1;
        for (i = 0; i < width; i++) {
            minus_state[i] += weight * (right * minus[i]);
        }
    }
    if (minus_active > 0) {
        for (i = 0; i < width; i++) {
            minus_state[i] /= (double)minus_active;
        }
    }
    return axis_quantity(plus_state, plus_active, plus, width)
        - axis_quantity(minus_state, minus_active, minus, width);
}

static void nearest_detail(
    double pred,
    const double *values,
    int n,
    int *best_out,
    double *margin_out,
    double *dist_out
) {
    int best = 0;
    double best_dist;
    double runner = 0.0;
    int seen_runner = 0;
    int i;
    best_dist = fabs(pred - values[0]);
    for (i = 1; i < n; i++) {
        double dist = fabs(pred - values[i]);
        if (dist < best_dist || (dist == best_dist && i < best)) {
            best = i;
            best_dist = dist;
        }
    }
    for (i = 0; i < n; i++) {
        double dist;
        if (i == best) {
            continue;
        }
        dist = fabs(pred - values[i]);
        if (!seen_runner || dist < runner) {
            runner = dist;
            seen_runner = 1;
        }
    }
    if (!seen_runner) {
        runner = best_dist;
    }
    *best_out = best;
    *margin_out = runner - best_dist;
    *dist_out = best_dist;
}

static int shed(
    double *quantity,
    double step,
    int cap,
    int minus_sign,
    const double *plus,
    const double *minus,
    int width,
    double collapse,
    double gate,
    double drop
) {
    int count = 0;
    while (count < cap) {
        double nxt = fsot_consensus_quantity(
            *quantity, step, minus_sign, plus, minus, width, collapse, gate
        );
        if (nxt < -drop) {
            break;
        }
        *quantity = nxt;
        count += 1;
    }
    return count;
}

EXPORT void fsot_read_place(
    double quantity,
    const double *gauges,
    int n,
    double ten,
    double hundred,
    double thousand,
    double ten_thousand,
    double hundred_thousand,
    double million,
    const double *plus,
    const double *minus,
    int width,
    double collapse,
    double gate,
    double drop,
    int minus_sign,
    int *named,
    double *remainder,
    double *margin,
    double *dist
) {
    int cap = n > 0 ? n - 1 : 0;
    int millions;
    int hundred_thousands;
    int ten_thousands;
    int thousands;
    int hundreds;
    int tens;
    int units = 0;
    double margin_v = 0.0;
    double dist_v = 0.0;
    millions = shed(
        &quantity, million, cap, minus_sign, plus, minus, width, collapse, gate, drop
    );
    hundred_thousands = shed(
        &quantity, hundred_thousand, cap, minus_sign, plus, minus, width, collapse, gate, drop
    );
    ten_thousands = shed(
        &quantity, ten_thousand, cap, minus_sign, plus, minus, width, collapse, gate, drop
    );
    thousands = shed(
        &quantity, thousand, cap, minus_sign, plus, minus, width, collapse, gate, drop
    );
    hundreds = shed(
        &quantity, hundred, cap, minus_sign, plus, minus, width, collapse, gate, drop
    );
    tens = shed(&quantity, ten, cap, minus_sign, plus, minus, width, collapse, gate, drop);
    if (n > 0 && gauges != NULL) {
        nearest_detail(quantity, gauges, n, &units, &margin_v, &dist_v);
    }
    if (named != NULL) {
        *named = ((((((millions * n + hundred_thousands) * n + ten_thousands) * n + thousands) * n + hundreds) * n + tens) * n + units);
    }
    if (remainder != NULL) {
        *remainder = quantity;
    }
    if (margin != NULL) {
        *margin = margin_v;
    }
    if (dist != NULL) {
        *dist = dist_v;
    }
}

/* Repeated consensus on one vessel. Same order as Lattice.fold_steps. */
EXPORT double fsot_fold_steps(
    double left,
    const int *signs,
    const double *rights,
    int n,
    const double *plus,
    const double *minus,
    int width,
    double collapse,
    double gate
) {
    double acc = left;
    int i;
    if (n <= 0 || signs == NULL || rights == NULL) {
        return left;
    }
    for (i = 0; i < n; i++) {
        acc = fsot_consensus_quantity(
            acc, rights[i], signs[i], plus, minus, width, collapse, gate
        );
    }
    return acc;
}

/* Gauges and place steps for one surface. The per-call attend stays unchanged. */
static int g_bound = 0;
static int g_width = 0;
static int g_n = 0;
static int g_minus_sign = -1;
static double g_collapse = 0.0;
static double g_gate = 0.0;
static double g_drop = 0.0;
static double g_ten = 0.0;
static double g_hundred = 0.0;
static double g_thousand = 0.0;
static double g_ten_thousand = 0.0;
static double g_hundred_thousand = 0.0;
static double g_million = 0.0;
static double g_plus[64];
static double g_minus[64];
static double g_gauges[16];

EXPORT int fsot_bind_read(
    const double *gauges,
    int n,
    double ten,
    double hundred,
    double thousand,
    double ten_thousand,
    double hundred_thousand,
    double million,
    const double *plus,
    const double *minus,
    int width,
    double collapse,
    double gate,
    double drop,
    int minus_sign
) {
    int i;
    g_bound = 0;
    if (gauges == NULL || plus == NULL || minus == NULL) {
        return 0;
    }
    if (n <= 0 || n > 16 || width <= 0 || width > 64) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        g_gauges[i] = gauges[i];
    }
    for (i = 0; i < width; i++) {
        g_plus[i] = plus[i];
        g_minus[i] = minus[i];
    }
    g_n = n;
    g_width = width;
    g_ten = ten;
    g_hundred = hundred;
    g_thousand = thousand;
    g_ten_thousand = ten_thousand;
    g_hundred_thousand = hundred_thousand;
    g_million = million;
    g_collapse = collapse;
    g_gate = gate;
    g_drop = drop;
    g_minus_sign = minus_sign;
    g_bound = 1;
    return 1;
}

/* One crossing: consensus, then the place read, on the bound surface. */
EXPORT void fsot_consensus_read(
    double left,
    double right,
    int sign,
    double *quantity,
    int *named,
    double *remainder,
    double *margin,
    double *dist
) {
    double value = 0.0;
    if (g_bound) {
        value = fsot_consensus_quantity(
            left, right, sign, g_plus, g_minus, g_width, g_collapse, g_gate
        );
        fsot_read_place(
            value,
            g_gauges,
            g_n,
            g_ten,
            g_hundred,
            g_thousand,
            g_ten_thousand,
            g_hundred_thousand,
            g_million,
            g_plus,
            g_minus,
            g_width,
            g_collapse,
            g_gate,
            g_drop,
            g_minus_sign,
            named,
            remainder,
            margin,
            dist
        );
    } else {
        if (named != NULL) {
            *named = 0;
        }
        if (remainder != NULL) {
            *remainder = 0.0;
        }
        if (margin != NULL) {
            *margin = 0.0;
        }
        if (dist != NULL) {
            *dist = 0.0;
        }
    }
    if (quantity != NULL) {
        *quantity = value;
    }
}
