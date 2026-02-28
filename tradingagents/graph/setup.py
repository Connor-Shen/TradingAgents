# TradingAgents/graph/setup.py

from typing import Dict, Optional, Set

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    DEFAULT_FIXED_TEAMS = {"research", "trading", "risk", "portfolio"}
    VALID_FIXED_TEAMS = {"research", "trading", "risk", "portfolio"}

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic

    def _normalize_fixed_teams(
        self, enabled_fixed_teams: Optional[list[str]]
    ) -> Set[str]:
        """Normalize and validate optional fixed-team keys."""
        if enabled_fixed_teams is None:
            return set(self.DEFAULT_FIXED_TEAMS)

        normalized = {
            team.lower().strip()
            for team in enabled_fixed_teams
            if isinstance(team, str) and team.strip()
        }
        valid = normalized.intersection(self.VALID_FIXED_TEAMS)
        return valid

    @staticmethod
    def _build_investment_plan_from_reports(state: dict) -> str:
        """Build a deterministic investment-plan fallback from analyst reports."""
        sections = []
        for title, key in (
            ("Market Analysis", "market_report"),
            ("Social Sentiment", "sentiment_report"),
            ("News Analysis", "news_report"),
            ("Fundamentals Analysis", "fundamentals_report"),
        ):
            report = (state.get(key) or "").strip()
            if report:
                sections.append(f"## {title}\n{report}")

        if not sections:
            return (
                "Auto-generated investment plan because Research Team is disabled.\n\n"
                "No analyst reports were available. Provisional stance: HOLD."
            )

        return (
            "Auto-generated investment plan because Research Team is disabled.\n\n"
            + "\n\n".join(sections)
            + "\n\nProvisional stance: HOLD unless clear BUY/SELL evidence appears above."
        )

    @staticmethod
    def _create_investment_plan_synthesizer():
        def node(state) -> dict:
            return {
                "investment_plan": GraphSetup._build_investment_plan_from_reports(state),
            }

        return node

    @staticmethod
    def _create_trader_passthrough():
        def node(state) -> dict:
            plan = (state.get("investment_plan") or "").strip()
            if not plan:
                plan = GraphSetup._build_investment_plan_from_reports(state)
            return {
                "trader_investment_plan": (
                    "Trading Team disabled; forwarding investment plan as trader plan.\n\n"
                    + plan
                )
            }

        return node

    @staticmethod
    def _create_final_decision_passthrough(reason: str):
        def node(state) -> dict:
            risk_history = (
                state.get("risk_debate_state", {}).get("history", "") or ""
            ).strip()
            trader_plan = (state.get("trader_investment_plan") or "").strip()
            investment_plan = (state.get("investment_plan") or "").strip()

            body = risk_history or trader_plan or investment_plan
            if not body:
                body = "FINAL TRANSACTION PROPOSAL: **HOLD**"

            return {
                "final_trade_decision": (
                    f"[Auto Final Decision: {reason}]\n\n{body}"
                )
            }

        return node

    def setup_graph(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        enabled_fixed_teams: Optional[list[str]] = None,
    ):
        """Set up and compile the agent workflow graph."""
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        enabled_fixed = self._normalize_fixed_teams(enabled_fixed_teams)

        enable_research = "research" in enabled_fixed
        enable_trading = "trading" in enabled_fixed
        enable_risk = "risk" in enabled_fixed
        enable_portfolio = "portfolio" in enabled_fixed and enable_risk

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(self.quick_thinking_llm)
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(self.quick_thinking_llm)
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}",
                delete_nodes[analyst_type],
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add fixed-team nodes conditionally
        if enable_research:
            workflow.add_node(
                "Bull Researcher",
                create_bull_researcher(self.quick_thinking_llm, self.bull_memory),
            )
            workflow.add_node(
                "Bear Researcher",
                create_bear_researcher(self.quick_thinking_llm, self.bear_memory),
            )
            workflow.add_node(
                "Research Manager",
                create_research_manager(self.deep_thinking_llm, self.invest_judge_memory),
            )
        else:
            workflow.add_node(
                "Investment Plan Synthesizer",
                self._create_investment_plan_synthesizer(),
            )

        if enable_trading:
            workflow.add_node(
                "Trader",
                create_trader(self.quick_thinking_llm, self.trader_memory),
            )
        else:
            workflow.add_node("Trader Passthrough", self._create_trader_passthrough())

        if enable_risk:
            workflow.add_node(
                "Aggressive Analyst",
                create_aggressive_debator(self.quick_thinking_llm),
            )
            workflow.add_node(
                "Neutral Analyst",
                create_neutral_debator(self.quick_thinking_llm),
            )
            workflow.add_node(
                "Conservative Analyst",
                create_conservative_debator(self.quick_thinking_llm),
            )
            if enable_portfolio:
                workflow.add_node(
                    "Risk Judge",
                    create_risk_manager(self.deep_thinking_llm, self.risk_manager_memory),
                )
            else:
                workflow.add_node(
                    "Final Decision Passthrough",
                    self._create_final_decision_passthrough("Portfolio Team disabled"),
                )
        else:
            workflow.add_node(
                "Final Decision Passthrough",
                self._create_final_decision_passthrough("Risk Team disabled"),
            )

        # Define edges
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        after_analysts_node = (
            "Bull Researcher" if enable_research else "Investment Plan Synthesizer"
        )
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, after_analysts_node)

        # Connect research stage
        trading_entry = "Trader" if enable_trading else "Trader Passthrough"
        if enable_research:
            workflow.add_conditional_edges(
                "Bull Researcher",
                self.conditional_logic.should_continue_debate,
                {
                    "Bear Researcher": "Bear Researcher",
                    "Research Manager": "Research Manager",
                },
            )
            workflow.add_conditional_edges(
                "Bear Researcher",
                self.conditional_logic.should_continue_debate,
                {
                    "Bull Researcher": "Bull Researcher",
                    "Research Manager": "Research Manager",
                },
            )
            workflow.add_edge("Research Manager", trading_entry)
        else:
            workflow.add_edge("Investment Plan Synthesizer", trading_entry)

        # Connect trading/risk/final stages
        if enable_risk:
            workflow.add_edge(trading_entry, "Aggressive Analyst")
            risk_end_target = "Risk Judge" if enable_portfolio else "Final Decision Passthrough"

            workflow.add_conditional_edges(
                "Aggressive Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Conservative Analyst": "Conservative Analyst",
                    "Risk Judge": risk_end_target,
                },
            )
            workflow.add_conditional_edges(
                "Conservative Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Neutral Analyst": "Neutral Analyst",
                    "Risk Judge": risk_end_target,
                },
            )
            workflow.add_conditional_edges(
                "Neutral Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Aggressive Analyst": "Aggressive Analyst",
                    "Risk Judge": risk_end_target,
                },
            )

            workflow.add_edge(risk_end_target, END)
        else:
            workflow.add_edge(trading_entry, "Final Decision Passthrough")
            workflow.add_edge("Final Decision Passthrough", END)

        return workflow.compile()
