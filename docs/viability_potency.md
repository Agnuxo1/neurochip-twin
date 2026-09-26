# G2b: independent viability anchors and potency references

## Rebuild

From the workspace root, with the local dependencies already installed:

```powershell
$env:PYTHONPATH = 'data/.pylibs;repo/src'
python -m neurotwin.data.dntdiver
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest repo/tests/test_dntdiver.py -q -p no:cacheprovider --basetemp data/processed/epa_nfa/pytest_tmp
```

The builder reads only local source files. Their SHA256 values and coverage are
recorded in `repo/results/viability_join_audit.json` and
`repo/results/reference_potency_audit.json`. Output tables are
`data/processed/epa_nfa/viability.parquet` and
`data/processed/epa_nfa/reference_potency.parquet`.

## Viability resolution and rules

PubChem CCTE_Shafer_MEA_dev AID **2284083** is viability AB; AID **2284068** is
viability LDH. The export specifies AC50 and BMD in **micromolar** and defines
`HITC >= 0.90` as active. Several PubChem substance IDs map to the same DTXSID.
We collapse them to one DTXSID × AID row only when their binary calls agree.
Discordant calls have `hit_call = null` and `hit_call_conflict = true`. The
reported AC50 is the median of positive AC50 values only when all source SIDs
are active; min/max and source SID list preserve the ambiguity. An inactive
curve may still have a fitted AC50 in the export; it is **not** an activity
potency in this table.

NTP DNT-DIVER 2018 `USEPA 91 neuron firing` viability rows remain one record
per source measurement, including their NTP plate, row, column, concentration,
raw response, and NTP normalized response. NTP chemical CASRN maps to DTXSID
through the release's `List_of_Chemicals.xlsx`. These are **NTP wells from a
different experiment**. The linkage to NFA is by chemical identity only; it
never identifies an NFA well or NFA plate. NTP's release README also warns
that one well can have multiple measurements.

## Potency references and endpoint mapping

The pinned EPA NFA DIV12 and AUC `tcplfit2` tables provide per-chemical
published fits for 7 and 17 NFA endpoints, respectively. These are reference
fits independent of our prediction model, but derived from the NFA assay and
therefore **not** an external assay validation. Preserve the original active
call (`hitc >= 0.90`) and numeric fitted values; only active calls with positive
finite BMD/AC50 are usable as potency references. DIV12 and AUC are separate
analyses, not independent replicate experiments.

The NTP Hill and Curvep BMCs are an external `USEPA 91 neuron firing` assay.
Their BMD/BMC, BMDL and BMDU are in **micromolar**; `bmd_log10_uM` is the base-10
logarithm of the numeric micromolar value. NTP inactive fits or BMD outside
the assay's active criterion must not be treated as positive potency labels.
Nine NTP endpoint names map to the canonical features in
`neurotwin.models.trajectory.FEATURES`:

| NTP endpoint | NFA feature |
|---|---|
| bursts per minute | burst_rate |
| mean correlation | correlation_coefficient_mean |
| mean firing rate | firing_rate_mean |
| mutual information | mutual_information_norm |
| number active electrodes | active_electrodes_number |
| number actively bursting electrodes | bursting_electrodes_number |
| number network spikes | network_spike_number |
| percent spikes in burst | per_burst_spike_percent |
| percent spikes in network spike | per_network_spike_spike_percent |

The other eight NFA features have no direct endpoint in this NTP protocol.
No name-only chemical matching or parent/salt collapsing is performed.

Sources: local PubChem AID exports (`data/neurotox_probe/aid*.csv`),
the [NTP DNT-DIVER 2018 release](https://doi.org/10.22427/NTP-DATA-002-00062-0001-0000-1)
and the pinned [EPA NFA refinement repository](https://github.com/USEPA/CompTox-DNT-NFA-Refinement/tree/01adf3e1a0068c87fe221d60df36b9f96c4b4b1d).
