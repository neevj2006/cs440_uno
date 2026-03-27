package src.pas.uno.agents;

// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class ExpectedOutcomeAgent extends MCTSAgent {

    public static class MCTSNode extends Node {
        public MCTSNode(final GameView game, final int logicalPlayerIdx, final Node parent) {
            super(game, logicalPlayerIdx, parent);
        }

        @Override
        public Node getChild(final Move move) {
            Game childGame = new Game(this.getGameView());
            childGame.setListener(null);
            injectDummyAgents(childGame);

            if (move != null) {
                childGame.resolveMove(move);
            } else {
                NodeState state = this.getNodeState();
                if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                    int drawCount = childGame.getUnresolvedCards().total();
                    childGame.drawTotal(childGame.getHand(this.getLogicalPlayerIdx()), drawCount);
                    childGame.getUnresolvedCards().clear();
                    childGame.getPlayerOrder().advance();
                } else {
                    childGame.getPlayerOrder().advance();
                }
            }
            return new MCTSNode(childGame.getOmniscientView(), childGame.getPlayerOrder().getCurrentLogicalPlayerIdx(),
                    this);
        }
    }

    public ExpectedOutcomeAgent(final int playerIdx, final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {
        MCTSNode root = new MCTSNode(game, this.getLogicalPlayerIdx(), null);

        int numActions = 0;
        Node.NodeState state = root.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            numActions = root.getOrderedLegalMoves().size();
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            numActions = 2;
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            numActions = 1;
        }

        if (numActions == 0)
            return root;

        long endTime = System.currentTimeMillis() + this.getMaxThinkingTimeInMS() - 150;

        Node[] children = new Node[numActions];
        try {
            for (int i = 0; i < numActions; i++) {
                Move move = getMoveForActionIdx(root, i, drawnCardIdx);
                children[i] = root.getChild(move);
            }
        } catch (Exception e) {
            return root;
        }

        int maxDepth = 1;
        while (!Thread.currentThread().isInterrupted() && System.currentTimeMillis() < endTime) {
            for (int actionIdx = 0; actionIdx < numActions; actionIdx++) {
                if (System.currentTimeMillis() >= endTime || Thread.currentThread().isInterrupted())
                    break;

                try {
                    Game childGame = new Game(children[actionIdx].getGameView());
                    childGame.setListener(null);
                    injectDummyAgents(childGame);

                    float qValue = alphaBeta(childGame, maxDepth, -1000.0f, 1000.0f, endTime);

                    root.setQValueTotal(actionIdx, root.getQValueTotal(actionIdx) + qValue);
                    root.setQCount(actionIdx, root.getQCount(actionIdx) + 1);
                } catch (Exception e) {
                    break;
                }
            }
            maxDepth++;
        }

        return root;
    }

    @Override
    public Move argmaxQValues(final Node node) {
        Node.NodeState state = node.getNodeState();
        int numActions = 0;

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            numActions = node.getOrderedLegalMoves().size();
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            numActions = 2;
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            numActions = 1;
        }

        if (numActions == 0)
            return null;

        int bestIdx = 0;
        float maxQ = -Float.MAX_VALUE;

        for (int i = 0; i < numActions; i++) {
            float q = node.getQValue(i);
            if (q > maxQ) {
                maxQ = q;
                bestIdx = i;
            }
        }

        return getMoveForActionIdx(node, bestIdx, null);
    }

    // ==========================================================
    // MINIMAX / ALPHA-BETA CORE ALGORITHM
    // ==========================================================

    private float alphaBeta(Game game, int depth, float alpha, float beta, long endTime) {
        if (System.currentTimeMillis() >= endTime || Thread.currentThread().isInterrupted()) {
            return 0.0f;
        }
        if (game.isOver()) {
            return evaluateTerminalState(game);
        }
        if (depth == 0) {
            return runRollouts(game, 3, endTime);
        }

        int currLogicalIdx = game.getPlayerOrder().getCurrentLogicalPlayerIdx();
        boolean isMaximizing = (currLogicalIdx == this.getLogicalPlayerIdx());

        List<Game> nextStates = generateNextStates(game, currLogicalIdx);

        if (isMaximizing) {
            float maxEval = -1000.0f;
            for (Game childState : nextStates) {
                float eval = alphaBeta(childState, depth - 1, alpha, beta, endTime);
                maxEval = Math.max(maxEval, eval);
                alpha = Math.max(alpha, eval);
                if (beta <= alpha)
                    break;
            }
            return maxEval;
        } else {
            float minEval = 1000.0f;
            for (Game childState : nextStates) {
                float eval = alphaBeta(childState, depth - 1, alpha, beta, endTime);
                minEval = Math.min(minEval, eval);
                beta = Math.min(beta, eval);
                if (beta <= alpha)
                    break;
            }
            return minEval;
        }
    }

    // ==========================================================
    // HELPER METHODS
    // ==========================================================

    private boolean isCardWild(Card c) {
        if (c == null)
            return false;
        if (c.isWild())
            return true;
        if (c.value() == Value.WILD || c.value() == Value.WILD_DRAW_FOUR)
            return true;
        return false;
    }

    private Color getValidRandomColor() {
        Color[] validColors = { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW };
        return validColors[this.getRandom().nextInt(validColors.length)];
    }

    private List<Game> generateNextStates(Game game, int logicalIdx) {
        List<Game> states = new ArrayList<>();
        try {
            if (game.getUnresolvedCards().total() > 0) {
                Game next = new Game(game.getOmniscientView());
                next.setListener(null);
                injectDummyAgents(next);

                int drawCount = next.getUnresolvedCards().total();
                next.drawTotal(next.getHand(logicalIdx), drawCount);
                next.getUnresolvedCards().clear();
                next.getPlayerOrder().advance();
                states.add(next);
                return states;
            }

            Set<Integer> legalMoves = game.getOmniscientView().getHandView(logicalIdx)
                    .getLegalMoves(game.getOmniscientView());
            if (!legalMoves.isEmpty()) {
                for (int moveIdx : legalMoves) {
                    Card c = game.getOmniscientView().getHandView(logicalIdx).getCard(moveIdx);

                    if (isCardWild(c)) {
                        Color[] colors = { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW };
                        for (Color color : colors) {
                            Game next = new Game(game.getOmniscientView());
                            next.setListener(null);
                            injectDummyAgents(next);

                            Agent nextAgent = next.getAgent(logicalIdx);
                            Move m = Move.createMove(nextAgent, moveIdx, color);
                            next.resolveMove(m);
                            states.add(next);
                        }
                    } else {
                        Game next = new Game(game.getOmniscientView());
                        next.setListener(null);
                        injectDummyAgents(next);

                        Agent nextAgent = next.getAgent(logicalIdx);
                        Move m = Move.createMove(nextAgent, moveIdx);
                        next.resolveMove(m);
                        states.add(next);
                    }
                }
            } else {
                Game nextKeep = new Game(game.getOmniscientView());
                nextKeep.setListener(null);
                injectDummyAgents(nextKeep);

                nextKeep.drawCard(nextKeep.getHand(logicalIdx));
                GameView postDrawView = nextKeep.getOmniscientView();
                int drawnIdx = postDrawView.getHandView(logicalIdx).size() - 1;
                Set<Integer> postDrawLegal = postDrawView.getHandView(logicalIdx).getLegalMoves(postDrawView);

                if (postDrawLegal.contains(drawnIdx)) {
                    Card c = postDrawView.getHandView(logicalIdx).getCard(drawnIdx);
                    if (isCardWild(c)) {
                        Color[] colors = { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW };
                        for (Color color : colors) {
                            Game nextPlay = new Game(nextKeep.getOmniscientView());
                            nextPlay.setListener(null);
                            injectDummyAgents(nextPlay);

                            Agent nextAgent = nextPlay.getAgent(logicalIdx);
                            Move m = Move.createMove(nextAgent, drawnIdx, color);
                            nextPlay.resolveMove(m);
                            states.add(nextPlay);
                        }
                    } else {
                        Game nextPlay = new Game(nextKeep.getOmniscientView());
                        nextPlay.setListener(null);
                        injectDummyAgents(nextPlay);

                        Agent nextAgent = nextPlay.getAgent(logicalIdx);
                        Move m = Move.createMove(nextAgent, drawnIdx);
                        nextPlay.resolveMove(m);
                        states.add(nextPlay);
                    }
                }
                nextKeep.getPlayerOrder().advance();
                states.add(nextKeep);
            }
        } catch (Exception e) {
        }
        return states;
    }

    private float runRollouts(Game state, int numRollouts, long endTime) {
        float totalUtility = 0.0f;
        for (int i = 0; i < numRollouts; i++) {
            if (System.currentTimeMillis() >= endTime)
                break;

            Game simGame = new Game(state.getOmniscientView());
            simGame.setListener(null);
            injectDummyAgents(simGame);
            totalUtility += playSingleRandomGame(simGame);
        }
        return totalUtility / Math.max(1, numRollouts);
    }

    private float playSingleRandomGame(Game simGame) {
        int maxDepth = 250;
        int depth = 0;

        while (!simGame.isOver() && depth < maxDepth && !Thread.currentThread().isInterrupted()) {
            int currLogicalIdx = simGame.getPlayerOrder().getCurrentLogicalPlayerIdx();
            GameView view = simGame.getOmniscientView();

            if (simGame.getUnresolvedCards().total() > 0) {
                int drawCount = simGame.getUnresolvedCards().total();
                simGame.drawTotal(simGame.getHand(currLogicalIdx), drawCount);
                simGame.getUnresolvedCards().clear();
                simGame.getPlayerOrder().advance();
            } else {
                Set<Integer> legalMovesSet = view.getHandView(currLogicalIdx).getLegalMoves(view);

                if (!legalMovesSet.isEmpty()) {
                    List<Integer> legal = new ArrayList<>(legalMovesSet);
                    int randMoveIdx = legal.get(this.getRandom().nextInt(legal.size()));
                    Card c = view.getHandView(currLogicalIdx).getCard(randMoveIdx);

                    Agent currAgent = simGame.getAgent(currLogicalIdx);
                    Move randomMove = isCardWild(c) ? Move.createMove(currAgent, randMoveIdx, getValidRandomColor())
                            : Move.createMove(currAgent, randMoveIdx);

                    simGame.resolveMove(randomMove);
                } else {
                    simGame.drawCard(simGame.getHand(currLogicalIdx));
                    GameView postDrawView = simGame.getOmniscientView();
                    Set<Integer> postDrawLegal = postDrawView.getHandView(currLogicalIdx).getLegalMoves(postDrawView);
                    int drawnCardIdx = postDrawView.getHandView(currLogicalIdx).size() - 1;

                    if (postDrawLegal.contains(drawnCardIdx) && this.getRandom().nextBoolean()) {
                        Card c = postDrawView.getHandView(currLogicalIdx).getCard(drawnCardIdx);
                        Agent currAgent = simGame.getAgent(currLogicalIdx);

                        Move randomMove = isCardWild(c)
                                ? Move.createMove(currAgent, drawnCardIdx, getValidRandomColor())
                                : Move.createMove(currAgent, drawnCardIdx);
                        simGame.resolveMove(randomMove);
                    } else {
                        simGame.getPlayerOrder().advance();
                    }
                }
            }
            depth++;
        }
        return evaluateTerminalState(simGame);
    }

    private float evaluateTerminalState(Game game) {
        for (int i = 0; i < game.getNumPlayers(); i++) {
            if (game.getOmniscientView().getHandView(i).size() == 0) {
                return (i == this.getLogicalPlayerIdx()) ? 1.0f : -1.0f;
            }
        }
        return 0.0f;
    }

    private Move getMoveForActionIdx(Node node, int actionIdx, Integer drawnCardIdx) {
        Node.NodeState state = node.getNodeState();
        int currLogicalIdx = node.getLogicalPlayerIdx();
        GameView view = node.getGameView();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            int cardIdx = node.getOrderedLegalMoves().get(actionIdx);
            Card c = view.getHandView(currLogicalIdx).getCard(cardIdx);

            if (isCardWild(c)) {
                return Move.createMove(this, cardIdx, getValidRandomColor());
            } else {
                return Move.createMove(this, cardIdx);
            }
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            if (actionIdx == 1) {
                int cIdx = (drawnCardIdx != null) ? drawnCardIdx : (view.getHandView(currLogicalIdx).size() - 1);
                Card c = view.getHandView(currLogicalIdx).getCard(cIdx);
                if (isCardWild(c)) {
                    return Move.createMove(this, cIdx, getValidRandomColor());
                } else {
                    return Move.createMove(this, cIdx);
                }
            }
        }
        return null;
    }

    public static void injectDummyAgents(Game game) {
        try {
            java.lang.reflect.Field agentsField = Game.class.getDeclaredField("agents");
            agentsField.setAccessible(true);
            Object currentAgents = agentsField.get(game);

            if (currentAgents == null) {
                Agent[] dummyAgents = new Agent[game.getNumPlayers()];
                for (int i = 0; i < dummyAgents.length; i++) {
                    int realIdx = game.getPlayerOrder().getAgentIdx(i);
                    ExpectedOutcomeAgent dummy = new ExpectedOutcomeAgent(realIdx, 100);
                    dummy.setRandom(new Random());
                    dummyAgents[i] = dummy;
                }
                agentsField.set(game, dummyAgents);
            }
        } catch (Exception e) {
        }
    }
}