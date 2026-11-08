# python/m4/m4.2_homework.py
"""M4.2 Homework: Give Each Subagent Its Own Scoped Scratch Folder.

THE IDEA
The lab's genre-researcher subagents each wrote raw search notes to their own
assigned /research/<genre>/ folder, kept out of the editor's context by
FilesystemPermission scoping: researchers could write under /research/**, the
editor could not.

This homework asks you to build a small team of 2 subagent types for
a domain YOU pick (e.g., trip planning, home renovation), each with
its own private, permission-scoped folder under /scratch/<name>/ to
stash raw notes in before answering. The harness below wires up the scratch
folder, the permissions, and the write-before-answering instruction for you;
you just decide who your two subagents are.

WHAT YOU FILL IN
  TODO 1: for each of the two entries in SUBAGENT_SPECS, fill in "name",
    "description" (when the main agent should call it), and "role_prompt"
    (who this subagent is and what its job is). Everything else -- the
    scratch folder, the permissions, the instruction to save raw notes
    before answering -- is handled for you.
  TODO 2: write the main agent's system prompt, telling it which subagent
    to call for which part of the job, and a user request that should
    trigger delegation to BOTH subagents.

RUN
  cd python
  uv run ./m4/m4.2_homework.py
"""

from deepagents import FilesystemPermission, create_deep_agent
from models import strong_model

SCRATCH_ROOT = "/scratch"


def scratch_path(subagent_name: str) -> str:
    return f"{SCRATCH_ROOT}/{subagent_name}/notes.md"


def scratch_permissions(subagent_name: str) -> list:
    """Scope a subagent to write only under its own scratch folder -- the same
    first-match-wins allow-then-deny pattern the lab used for
    research_permissions/editor_permissions."""
    return [
        FilesystemPermission(
            operations=["read", "write"],
            paths=[f"{SCRATCH_ROOT}/{subagent_name}/**"],
            mode="allow",
        ),
        FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
    ]


def scratch_instruction(subagent_name: str) -> str:
    return (
        f'Before you answer, call write_file on "{scratch_path(subagent_name)}" '
        "with your raw notes or reasoning. Then give your final answer using "
        "only the polished result -- do not repeat those raw notes in your reply."
    )


def build_subagents(specs: list[dict]) -> list[dict]:
    team = []
    for spec in specs:
        name = spec["name"]
        team.append(
            {
                "name": name,
                "description": spec["description"],
                "system_prompt": spec["role_prompt"]
                + "\n\n"
                + scratch_instruction(name),
                "permissions": scratch_permissions(name),
            }
        )
    return team


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Fill in your two subagents.
#
# For each entry: "name" is the handle the main agent calls it by, kebab-
# case (e.g. "flight-finder"). "description" is how the main agent decides
# which one to use. "role_prompt" is that subagent's own job description --
# don't mention scratch files or write_file here, that's added for you.
# ════════════════════════════════════════════════════════════════════════

SUBAGENT_SPECS = [
    {
        "name": "CurriculumBuilder",
        "description": "Call this subagent to bridge gaps in technical ML/AI skills, recommend specific frameworks (e.g., PyTorch Geometric, DeepChem), and outline hands-on portfolio projects.",
        "role_prompt": (
            "You are an expert AI for Drug Discovery (AIDD) Academic Advisor. Your job is to map out "
            "technical learning paths for someone who already knows chemistry and programming. "
            "Focus heavily on Geometric Deep Learning, Graph Neural Networks (GNNs) for molecular graphs, "
            "and transformers for SMILES strings. You MUST use your private scratch folder to parse syllabus data, "
            "compare open-source packages, and design portfolio project architectures BEFORE presenting a clean, "
            "prioritized learning roadmap to the main agent."
        ),
    },
    {
        "name": "IndustryHeadhunter",
        "description": "Call this subagent to identify target companies (TechBio/Pharma), analyze current job descriptions, translate resume skills, and design networking strategies.",
        "role_prompt": (
            "You are a specialized TechBio Talent Recruiter and Headhunter. Your job is to analyze the job market "
            "for AI Drug Discovery roles (e.g., Computational Chemist, ML Research Engineer). "
            "You focus on how to rebrand a traditional chemist/software background into a highly modern AI-driven profile. "
            "You MUST use your private scratch folder to cross-reference market requirements, map out networking angles, "
            "and draft tailored resume bullet points BEFORE delivering an actionable job placement strategy to the main agent."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write the main agent's system prompt and a triggering request.
#
# MAIN_PROMPT should tell the main agent about each subagent by name and
# when to call it (mirror how EDITOR_PROMPT in the lab named
# genre-researcher and explained the job).
# USER_REQUEST should be a task that should make the main agent delegate to
# BOTH of your subagents.
# ════════════════════════════════════════════════════════════════════════

MAIN_PROMPT = """You are a Master Executive Career Coach specializing in the TechBio and AI-driven pharmaceutical sectors. 
Your goal is to guide a highly qualified professional through a seamless pivot from a combined Chemistry & Software role into AI for Drug Discovery.

You have access to two specialized subagents:
1. CurriculumBuilder: Handles the specific technical upskilling, open-source tooling, and ML portfolio architecture.
2. IndustryHeadhunter: Handles market positioning, resume translation, and competitive corporate targeting.

When given a request, you must activate BOTH subagents to synthesize the technical and career development halves of the transition. 
Combine their findings into a cohesive, high-impact, 6-month Career Pivot Roadmap. Do not show internal scratchpad file paths to the user; deliver a polished, executive-ready master plan."""

USER_REQUEST = (
    "I am currently working as a hybrid Chemist and Software Developer, and I want to pivot my career completely "
    "into AI for Drug Discovery. I already understand molecular structures and basic software engineering, but "
    "I need a concrete plan. Please map out the exact ML/Deep Learning topics and libraries I need to learn to fill "
    "the gaps, design a high-impact portfolio project I can build to prove my skills, and outline how I should "
    "rebrand my existing dual background to catch the eye of top TechBio startups and pharma companies."
)

for _spec in SUBAGENT_SPECS:
    if _spec["name"].startswith("TODO-1"):
        raise NotImplementedError("TODO 1: see the comment block above")
if MAIN_PROMPT.startswith("TODO 2") or USER_REQUEST.startswith("TODO 2"):
    raise NotImplementedError("TODO 2: see the comment block above")

_team = build_subagents(SUBAGENT_SPECS)

# The main agent must never write into any subagent's scratch folder either.
MAIN_PERMISSIONS = [
    FilesystemPermission(
        operations=["write"], paths=[f"{SCRATCH_ROOT}/**"], mode="deny"
    ),
]

agent = create_deep_agent(
    model=strong_model,
    name="Homework_Team_Agent",
    system_prompt=MAIN_PROMPT,
    subagents=_team,
    permissions=MAIN_PERMISSIONS,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": USER_REQUEST}]},
    config={"recursion_limit": 50},
)
print(result["messages"][-1].text)

files = result.get("files", {})
print("\n--- Scratch folder isolation check ---")
for spec in SUBAGENT_SPECS:
    path = scratch_path(spec["name"])
    print(
        f"  {path}: {'found' if path in files else 'not written (subagent may not have been called)'}"
    )

scratch_files = [p for p in files if p.startswith(SCRATCH_ROOT + "/")]
expected = {scratch_path(spec["name"]) for spec in SUBAGENT_SPECS}
stray = [p for p in scratch_files if p not in expected]
if stray:
    print(f"  Unexpected scratch files (isolation may have failed): {stray}")
else:
    print("  No stray scratch files -- each subagent wrote only to its own folder.")
