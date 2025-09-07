# MASE Setup Guide for Claude Code

This guide walks you through setting up and running your first MASE (Many Agent Socratic Exploration) session using Claude Code.

## Quick Start

### 1. Clone or Fork This Repository
```bash
git clone https://github.com/m3untold/MASE.git
cd MASE
```

### 2. Create Your Agent Ensemble
The core MASE agents are defined in the `.claude/agents/` folder. You can use the provided agents or create your own.

**To use the standard MASE ensemble:**
- Copy the agent files from `.claude/agents/` to your Claude Code agents directory
- Or create agents using Claude Code's `/agents` command with the provided configurations

### 3. Start Your First Session
```
Let's begin a MASE session exploring [your topic]. Please invoke all seven agents in sequence to respond to this opening provocation:

"[Your complex question or scenario]"

Each agent should reference others and ask questions as they respond.
```

## Detailed Setup Instructions

### Prerequisites

**Required:**
- **Claude Account** - Active subscription to Claude (Pro or Team plan recommended for extended sessions)
- **Claude Code** - Anthropic's official CLI tool
- **Task Tool Access** - Ability to invoke specialized agents (included in Claude Code)

**Important Cost Considerations:**
- MASE sessions involve multiple agent invocations and extended conversations
- A typical 2-3 hour session may use significant message credits
- Consider Claude Pro or Team plans for regular MASE practice
- Sessions can be broken into shorter segments to manage costs

**Recommended:**
- Git for version control and sharing sessions
- Text editor with markdown support
- Basic familiarity with conversation facilitation

### Agent Configuration

Each MASE agent is defined with a specific configuration file:

```md
---
name: agent-name-here  
description: When and how to use this agent
model: sonnet
---

[Detailed persona and instructions]
```

#### The Standard 7-Agent Ensemble

1. **luma-child-voice** - 9-year-old demanding accessibility and moral clarity
2. **elowen-ecological-wisdom** - Indigenous knowledge and ceremonial practice  
3. **systems-analyst-orin** - Structural analysis and feedback patterns
4. **moral-imagination-explorer** - Creative reframing and speculative futures
5. **policy-pragmatist-sefi** - Governance realities and implementation
6. **capitalist-realist-tala** - Market constraints and institutional logic
7. **ilya-liminal-guide** - Metaphysical inquiry and threshold concepts

#### Creating Agents in Claude Code

**Method 1: Use the `/agents` command**
```
/agents create luma-child-voice
[Paste the agent configuration when prompted]
```

**Method 2: Copy provided configurations**
Copy the `.claude/agents/*.md` files to your Claude Code agents directory.

### Managing Costs and Session Length

**Cost Management Strategies:**
- **Start with shorter sessions** (60-90 minutes) to understand usage patterns
- **Break complex explorations** into multiple sessions rather than single marathon dialogues
- **Use strategic pausing** - document insights and continue later to avoid timeout costs
- **Focus agent responses** with specific questions rather than open-ended exploration
- **Consider group funding** for community MASE sessions to share costs

**Typical Usage Estimates:**
- **Initial agent responses** (7 agents): ~7-10 messages
- **Deep engagement round** (cross-referencing): ~7-14 messages  
- **Translation moments**: ~3-5 additional messages
- **Agent reflections**: ~7 messages
- **Total per session**: 25-40 messages depending on depth and length

### Session Structure Template

Use this basic structure for your MASE sessions:

```markdown
# MASE Session [Number]: [Topic Title]

**Date**: [Today's Date]
**Facilitator**: [Your Name]  
**Duration**: [Estimated Time]
**Theme**: [Core Question or Area]

---

## Opening Provocation

[Present your complex question or scenario here. Good provocations:]
- Have multiple valid perspectives
- Connect to real-world stakes  
- Allow for creative and analytical responses
- Can be understood by both children and experts

---

## Dialogue Evolution

[This is where the magic happens - agents will respond and build on each other]

---

## Key Insights Emerged

[Document the discoveries that emerged from agent interaction]

---

## Questions for Future Exploration

[What new questions arose? What deserves deeper inquiry?]
```

### Sample Session Starters

#### For Beginners
- "How do we balance individual freedom with collective responsibility?"
- "What makes a community truly inclusive?"
- "How should we prepare children for an uncertain future?"

#### For Intermediate Users  
- "What would governance look like if designed by those most affected by its decisions?"
- "How might we organize economic systems to serve life rather than growth?"
- "What does healing look like at the scale of communities and cultures?"

#### For Advanced Practice
- "How do we navigate the metacrisis without falling into either apocalyptic despair or naive optimism?"
- "What would it mean to design technology that serves collective flourishing rather than individual engagement?"
- "How might we create institutions that can hold complexity without becoming paralyzed by it?"

## Running Your First Session

### Step-by-Step Process

1. **Choose Your Provocation**
   - Start with something you genuinely want to explore
   - Ensure it's complex enough for multiple perspectives
   - Consider what stakes are involved for different groups

2. **Open the Dialogue**
   ```
   Let's begin a MASE session exploring [your topic]. 

   Opening Provocation: "[Your question]"

   Please have each of the seven agents respond in sequence: Luma, Elowen, Orin, Nyra, Sefi, Tala, and Ilya. Each should reference others and ask questions.
   ```

3. **Facilitate Cross-Engagement**
   - Watch for agents referencing each other
   - If they're not engaging, prompt: "Let's dive deeper into the tensions between [Agent A] and [Agent B]"
   - Encourage building on others' insights

4. **Translation Check**
   - Bring in Luma when concepts get too abstract
   - Ask: "Can you explain this in language a 9-year-old would understand?"

5. **Explore Tensions**
   - Don't resolve disagreements too quickly
   - Ask: "What's the productive tension here?"
   - Push into areas of uncertainty or confusion

6. **Document and Reflect**
   - Save the complete dialogue
   - Have agents write reflection journal entries
   - Note insights that surprised you

### Common Beginner Mistakes

❌ **Don't**: Let agents give sequential monologues without referencing each other
✅ **Do**: Require agents to build on, challenge, or question others' contributions

❌ **Don't**: Try to resolve all tensions and reach consensus  
✅ **Do**: Let productive disagreements generate new insights

❌ **Don't**: Keep conversations purely abstract
✅ **Do**: Ground insights in real experience and accessible language

❌ **Don't**: Rush through responses
✅ **Do**: Allow time for depth and genuine engagement between perspectives

## Advanced Facilitation Techniques

### Meta-Circles
Pause the main dialogue to reflect on process:
```
Let's take a meta-circle. Agents, please reflect:
- What patterns are emerging in our conversation?
- Where do we feel stuck or repetitive?
- What perspectives might we be missing?
- How is this dialogue changing your thinking?
```

### Translation Protocols
When complex concepts emerge:
```
Luma, you've been listening to this discussion about [complex concept]. 
Can you ask the questions a 9-year-old would ask? What doesn't make sense to you?

[Agent name], can you translate your insight into language Luma could understand?
```

### Tension Deep-Dives
When productive disagreements emerge:
```
I notice tension between [Agent A's] perspective on [topic] and [Agent B's] approach. 
Let's explore this tension more deeply. Other agents, what do you see in this disagreement? 
What might both be seeing that the other is missing?
```

### Temporal Integration
Expand time horizons:
```
Let's bring in multiple timeframes:
- Elowen, what would our ancestors say about this?
- Agents, how does this affect people right now?  
- What will children inherit from the decisions we're discussing?
```

## Documentation Best Practices

### Session Records
- **File naming**: `session_[number]_[topic_keywords].md`
- **Preserve verbatim**: Keep actual agent language and voice
- **Note paradigm shifts**: Track when thinking fundamentally changes
- **Include process insights**: What did the dialogue method itself reveal?

### Agent Reflection Journals
Create ongoing journals for each agent at:
`/agents/reflections/[agent-name]-reflections.md`

Template:
```md
### Session [Number] – [Title]

**What moved me:**  
...

**What I'm still questioning:**  
...

**New tensions I felt:**  
...

**What I want to remember next time:**  
...
```

### Insight Tracking
Maintain a running log of:
- **Emergent concepts**: New ideas that arose from interaction
- **Translation breakthroughs**: When complex ideas became accessible
- **Paradigm shifts**: When agents fundamentally changed perspective
- **Action implications**: How insights might influence real-world decisions

## Budget-Friendly MASE Practice

### For Individual Practitioners
- **Monthly MASE Sessions**: 1-2 sessions per month to explore ongoing questions
- **Micro-Sessions**: 30-45 minute focused explorations on specific tensions
- **Collaborative Sessions**: Share costs with friends/colleagues exploring similar topics
- **Documentation Focus**: Invest message credits in thorough documentation for future reference

### For Communities and Organizations
- **Group Subscriptions**: Share Claude Team accounts for community MASE practice
- **Workshop Model**: Intensive MASE sessions as periodic community events
- **Educational Partnerships**: Connect with institutions that might fund dialogue innovation
- **Grant Applications**: MASE methodology development may qualify for community/educational grants

### For Educators and Researchers
- **Classroom Integration**: Use MASE for complex topic exploration in courses
- **Research Applications**: MASE sessions as data collection method for collaborative inquiry
- **Student Projects**: Teams use MASE for thesis/capstone explorations
- **Faculty Development**: Departments explore pedagogical questions through MASE

## Troubleshooting Common Issues

### "My agents aren't engaging with each other"
- **Solution**: Be more explicit about requiring cross-referencing
- **Prompt**: "Before sharing your own view, please reference what [specific agent] just said"
- **Meta-intervention**: "Let's pause. I'm noticing agents aren't building on each other's ideas. How can we create more conversation?"

### "The dialogue is too abstract"
- **Solution**: Invoke Luma more frequently
- **Prompt**: "Luma, what questions would you ask about what you just heard?"
- **Grounding**: "How does this connect to specific people's lived experiences?"

### "We're going in circles"
- **Solution**: Introduce new provocations or shift perspectives  
- **Prompt**: "What are we not talking about that we should be?"
- **Meta-circle**: Have agents reflect on what's keeping them stuck

### "Agents are falling out of character"
- **Solution**: Provide more context in your prompts about the ongoing MASE session
- **Reminder**: "Remember this is MASE Session [X], building on insights from previous dialogues"
- **Reorientation**: Briefly remind agents of their established perspectives and relationships

### "The conversation lacks depth"
- **Solution**: Push into tensions rather than avoiding them
- **Prompt**: "Where do you feel most uncertain or confused about what others have said?"
- **Challenge**: "What assumptions are we making that we haven't examined?"

## Community and Sharing

### Contributing to MASE Development
- Fork the repository and share your session innovations
- Document new agent personas or specialized ensembles
- Share effective provocation templates and facilitation techniques
- Report issues or suggestions via GitHub issues

### Ethical Guidelines
- Respect agent autonomy and voice consistency
- Include diverse perspectives, especially those affected by topics discussed
- Share insights responsibly and credit collective process
- Preserve nuance and complexity when documenting sessions

### Finding Community
- Share your MASE experiments with the hashtag #MASEdialogue
- Join the Collective Futurecrafting community for related practices
- Connect with others exploring multi-agent dialogue methodologies
- Consider hosting community MASE sessions on shared topics

---

**Ready to begin?** Start with a question that genuinely intrigues you, invoke the agents, and let their wisdom surprise you. The most profound insights often emerge from unexpected directions.

**Need help?** Open an issue in the GitHub repository or connect with the MASE community. We're all learning how to do this together.