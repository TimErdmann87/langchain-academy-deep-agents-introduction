# python/m3/m3.2_homework.py
"""M3.2 Homework: Bundle a Reference File Into Your Skill.

THE IDEA
The lab's two skills (qualify-lead and draft-pitch) are each a single flat
SKILL.md file with everything inline. But the lesson also covered a third
stage of progressive disclosure: a skill can point to supporting files (a
reference doc, a template, a script) that live alongside SKILL.md and that
the agent only reads when it actually needs them, instead of stuffing
everything into the system prompt up front. This homework asks you to write
a skill for a topic or workflow YOU pick (not sales) that bundles a SECOND
file with details the agent needs but that aren't in SKILL.md itself, then
confirm from the trace that the agent actually called `read_file` on that
second file before answering, rather than guessing.

WHAT YOU FILL IN
  TODO 1: write your own SKILL.md content. It must instruct the agent to
    read a `reference.md` file (in the same skill directory) for specific
    details it needs, and must NOT restate those details inline. The `name`
    field in your frontmatter must exactly match SKILL_NAME below.
  TODO 2: write reference.md's content: the specific facts, numbers, or
    template your skill's instructions point to and depend on.
  TODO 3: write a system prompt and a user question that should activate
    your skill.

RUN
  cd python
  uv run ./m3/m3.2_homework.py
"""

import tempfile
from pathlib import Path
from textwrap import dedent

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from models import model

# This name becomes the skill's directory name. It must exactly match the
# `name:` field you write in the frontmatter inside build_skill_md() below.
SKILL_NAME = "submit-polymer-sample"
REFERENCE_PATH = f"/skills/{SKILL_NAME}/reference.md"


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own SKILL.md content.
#
# Requirements:
#   - YAML frontmatter with `name` (must equal SKILL_NAME above) and
#     `description` (a specific sentence describing WHEN to use this skill).
#   - Steps that tell the agent to open `reference.md` (in this same skill
#     directory) for the specific details it needs to do the task well.
#   - Do NOT put those details in SKILL.md itself; if the agent could do
#     the task correctly without ever reading reference.md, this doesn't
#     exercise progressive disclosure.
#
# Example shape (delete this and write your own):
#   return """---
#   name: your-skill-name
#   description: Use when the user wants to ...
#   ---
#
#   # Your Skill Title
#
#   **Step 1: ...**: ...
#   **Step 2: ...**: before proceeding, read reference.md in this skill's
#     directory for the exact ... to use. Do not guess these.
#   """
# ════════════════════════════════════════════════════════════════════════


def build_skill_md() -> str:
    """TODO 1: return your own SKILL.md content as a string."""
    skill_md_content = dedent(text="""\
      ---
      name: submit-polymer-sample
      description: Use when the user wants to submit a polymer sample for material characterization
      ---

      # Submit a polymer sample
      Use this skill when a chemist asks how to prepare, label, or submit a polymer
      sample for GPC, NMR, or DSC analysis, or asks when results will be ready.

      ## Important
      The Core's sample requirements, queue rules, drop-off windows, turnaround
      times, ID format, and submission form are facility-specific and change over
      time. They are NOT in this file. Before answering, you MUST read
      `reference.md` in this skill's directory and take every requirement, number,
      and deadline from it. Never answer from general chemistry knowledge or
      guess a value; if reference.md does not cover something, say so.

      ## Workflow
      **Step 1: Identify Analysis**: Clarify the analysis type(s) requested (GPC, NMR, DSC) and solubility of the polymer sample.
      **Step 2: Read 'reference.md'**: Use tool read_file to load reference information from 'reference.md' in this skill's directory.
      **Step 3: Check Requirements**: Look up the sample-prep requirements for that analysis and apply any conditional rules (e.g. molecular-weight or solubility-dependent ones) to this specific sample.
      **Step 4: Create SampleID***: Build the sample ID using the format in reference.md.
      **Step 5: Inform about Results Date***: Work out the expected results date from the drop-off rules and turnaround times, counting from when the user plans to drop off.
      **Step 6: Fill out Submission Form**: Fill in the submission form template from reference.md.

      ## Answer format
      - Short prep checklist tailored to the sample
      - Sample ID
      - Expected results date, with one line on how you got it
      - The completed submission form
      """)

    return skill_md_content


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write reference.md's content.
#
# This should contain the specific facts your SKILL.md pointed to and
# depends on: a rubric, a set of numbers, a template, a checklist. Specific
# enough that an answer produced without reading it would visibly differ
# from one produced with it.
# ════════════════════════════════════════════════════════════════════════


def build_reference_md() -> str:
    """TODO 2: return the content of your skill's reference.md."""
    skill_ref_md_content = dedent(text="""\
      # Building 3 Characterization Core: Submission Reference
      Revision 2026-10. These rules override general lab practice.

      ## Drop-off
      - Drop-off window: Monday to Thursday, before 2:00 pm, Room 3-114.
      - Samples dropped off after 2:00 pm count as received the next business day.
      - Friday drop-offs are accepted but logged as received the following Monday.
      - Turnaround is counted in business days from the "received" day.

      ## Sample ID format
      `YYYYMMDD-INITIALS-NNN`, where the date is the drop-off date and NNN is the
      submitter's 3-digit sequence number for that day (starting at 001).
      Example: `20261030-TB-001`.

      ## GPC
      - Default eluent: THF, 2.0 mg/mL, filtered through a 0.45 um PTFE syringe filter.
      - Polar polymers insoluble in THF (e.g. polyamides, PAN): DMF + 0.05 M LiBr,
        1.5 mg/mL, filtered through 0.45 um nylon.
      - Volume: 1.5 mL in a 2 mL amber vial, cap labeled with the sample ID.
      - Calibration: narrow polystyrene standards; results reported as
        PS-equivalent Mn, Mw, and dispersity.
      - Estimated Mn above 50 kDa: must be flagged "HMW" on the form; runs on the
        high-MW column set (PLgel Mixed-A).
      - Turnaround: 3 business days standard; 5 business days for HMW samples.

      ## NMR (400 MHz)
      - 10-15 mg dissolved in 0.6 mL CDCl3 (or DMSO-d6 if insoluble in CDCl3).
      - Standard 5 mm tube, filled to 4 cm.
      - Turnaround: 1H 1 business day; 13C 2 business days.

      ## DSC
      - 5-10 mg sealed in a Tzero aluminum pan (hermetic lid for volatile samples).
      - Default program: heat-cool-heat, -50 to 200 C, 10 C/min, N2 purge.
        Any other range must be written in the "Special instructions" field.
      - Turnaround: 4 business days.

      ## Submission form template
      ```
      CHARACTERIZATION CORE SUBMISSION
      Sample ID:
      Submitter / Group:
      Analysis:            [ ] GPC  [ ] NMR  [ ] DSC
      Polymer:
      Est. Mn (kDa):
      HMW flag (GPC):      [ ] Yes  [ ] No
      Solvent / Eluent:
      Concentration:
      Drop-off date:
      Expected results:
      Special instructions:
      ```
      """)

    return skill_ref_md_content


# Write the skill to a scratch directory so it's discoverable through a
# FilesystemBackend, the same mechanism the lab uses for python/m3/skills/.
_tmp_root = Path(tempfile.mkdtemp(prefix="m3_2_homework_"))
_skill_dir = _tmp_root / "skills" / SKILL_NAME
_skill_dir.mkdir(parents=True, exist_ok=True)
(_skill_dir / "SKILL.md").write_text(build_skill_md())
(_skill_dir / "reference.md").write_text(build_reference_md())

backend = FilesystemBackend(root_dir=str(_tmp_root), virtual_mode=True)
print(f"Skill files written to: {_skill_dir}")


# ════════════════════════════════════════════════════════════════════════
# TODO 3: Write a system prompt and a triggering question.
#
# SYSTEM_PROMPT: give the agent a persona of your choosing (a name, a
# voice, anything you want).
# USER_QUESTION: a question that should match your skill's `description`
# closely enough that the agent activates it.
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a lab assistant for a polymer chemistry research group.
You have access to skills and a read_file tool. When a request matches a 
skill, follow that skill's instructions, and read any supporting files it
points to instead of relying on general knowledge. Be concise and practical."""
USER_QUESTION = """I made a polystyrene-b-PMMA block copolymer, estimated Mn around
80 kDa. I want GPC on it and plan to drop it off this Friday morning
(Oct 30, 2026). My initials are TE and I work in the group 'Accelerated Discovery' 
and it's my only sample that day. How should I prepare it, and when will I get 
results?"""

agent = create_deep_agent(
    model=model,
    name="Homework_Agent",
    backend=backend,
    skills=["/skills"],
    system_prompt=SYSTEM_PROMPT,
)

result = agent.invoke({"messages": [{"role": "user", "content": USER_QUESTION}]})
print(result["messages"][-1].text)

read_calls = [
    call
    for msg in result["messages"]
    for call in getattr(msg, "tool_calls", [])
    if call["name"] == "read_file"
]
reference_was_read = any(
    call["args"].get("file_path") == REFERENCE_PATH for call in read_calls
)
print(f"\n--- Did the agent read {REFERENCE_PATH}? {reference_was_read} ---")
if not reference_was_read:
    print(
        "It didn't. Either SKILL.md isn't clearly telling it to, or the "
        "task is answerable without the details in reference.md."
    )
