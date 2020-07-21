import { css } from "lit-element";
import { updatePythonLanguageDefinition } from "./pythonTextCompletion";
import { monaco } from "./monaco";
import { sleep } from './utils';
import { PythonExecutor } from "./pythonExecutor";
import { History } from "./history";
import { promiseFromDomEvent } from "./utils"

updatePythonLanguageDefinition(monaco);

class ReplElement extends HTMLElement {
	static get defaultEditorOptions(){
		return {
			value: "\n",
			language: "python",
			folding : false,
			theme : "vs-dark",
			roundedSelection : false,
			fontSize : 20,
			minimap: {
				enabled: false
			},
			wordWrap: 'wordWrapColumn',
			wordWrapColumn: 80,
			overviewRulerLanes : 0,
			scrollBeyondLastLine : true,
			contextmenu : false
		};
    }
    
    static get styles() {
        return css`
            .root {
                height : 100%;
                width : 100%;
            }
        `
	}

	get startOfInputPosition(){
		return new monaco.Position(this.readOnlyLines + 1, 1);
	}

	get endOfInputPosition(){
		return this.editor.getModel().getPositionAt(this.editor.getValue().length);
	}

	get endOfInputSelection(){
		return monaco.Selection.fromPositions(this.endOfInputPosition, this.endOfInputPosition);
	}

	get allOfInputSelection(){
		return monaco.Selection.fromPositions(this.startOfInputPosition, this.endOfInputPosition);
	}

	get value(){
		return this.editor.getModel().getValueInRange(this.allOfInputSelection);
	}

	doesSelectionIntersectReadOnlyRegion(sel){
		return this.editor.getSelection().getStartPosition().isBefore(this.startOfInputPosition);
	}

	doesCurrentSelectionIntersectReadOnlyRegion(){
		return this.doesSelectionIntersectReadOnlyRegion(this.editor.getSelection());
	}

	atStartOfInputRegion(){
		return this.editor.getSelection().isEmpty() && this.editor.getPosition().equals(this.startOfInputPosition);
	}

	setSelectionDirection(sel, dir){
		let {lineNumber : sline, column : scol} = sel.getStartPosition();
		let {lineNumber : eline, column : ecol} = sel.getEndPosition();
		return monaco.Selection.createWithDirection(sline, scol, eline, ecol, dir);
	}

	get lineHeight(){
		return Number(this.querySelector(".view-line").style.height.slice(0,-2));
	}

	get screenHeight(){
		return Number(getComputedStyle(document.querySelector("repl-terminal")).height.slice(0,-2));
	}

	get linesPerScreen(){
		return Math.floor(this.screenHeight / this.lineHeight);
	}

	getScreenLinesInModelLine(lineNumber){
		let bottom = this.editor.getTopForPosition(lineNumber, this.editor.getModel().getLineLength(lineNumber));
		let top = this.editor.getTopForLineNumber(lineNumber);
		return Math.floor((bottom - top)/this.lineHeight + 1);
	}
	
	getLastScreenLineOfModelLine(lineNumber){
		let bottom = this.editor.getTopForPosition(lineNumber, this.editor.getModel().getLineLength(lineNumber));
		let top = 0;
		return Math.floor((bottom - top)/this.lineHeight + 1);
	}

	getScreenLineOfPosition(pos){
		let {lineNumber, column} = pos;
		if(!Number.isInteger(lineNumber) || !Number.isInteger(column)){
			throw Error("Invalid position!");
		}
		let bottom = this.editor.getTopForPosition(lineNumber, column);
		let top = 0;
		return Math.floor((bottom - top)/this.lineHeight + 1);
	}

	getModelLineCount(){
		return this.editor.getModel().getLineCount();
	}

	getScreenLineCount(){
		let totalLines = this.editor.getModel().getLineCount();
		return this.getLastScreenLineOfModelLine(totalLines);
	}

	revealSelection(scrollType = monaco.editor.ScrollType.Immediate){
		this.editor.revealRange(this.editor.getSelection(), scrollType);
	}

	focus(){
		this.editor.focus();
	}


	constructor(options){
		super();
		this._focused = false;
		this._visible = true;
		this.editorOptions = Object.assign(ReplElement.defaultEditorOptions, options);
		this.readOnlyLines = 1;
		this.readOnly = false;
        // window.addEventListener("pagehide", async (event) => {
        //     localStorage.setItem("pageHide", true);
        //     await this.history.writeToLocalStorage();
        // });
        this.executor = new PythonExecutor();
        this.history = new History();
		this.historyIdx = this.history.length || 0;
		this.historyIndexUndoStack = [];
		this.historyIndexRedoStack = [];
		this.firstLines = {};
		this.outputScreenLines = {};
		this.outputModelLines = {};
		this.outputModelLines[1] = true;
		this.outputScreenLines[1] = true;
		this.firstLines[2] = true; 
		this.lineOffsetMutationObserver = new MutationObserver((mutationsList, observer) => {
			this.updateLineOffsets();
		});
		// To display our output lines left of our input lines, we need to do the following:
		//    (1) remove contain:strict; and overflow:hide; from .lines-content
		//    (2) remove overflow:hide; from ".monaco-scrollable-element"
		// 	  (3) attach a mutation observer to .view-lines to make sure left-margin is set appropriately on output line content
		//    (4) attach a mutation observer to .view-overlays to make sure left-margin is set appropriately on output line overlays
		// We are doing these things here to ensure that everything continues to work when "setModel" is called.
		// Not clear whether this is necessary / a good idea...
		this.overflowMutationObserver = new MutationObserver((mutationsList, observer) => {
			for(let mutation of mutationsList){
				if(mutation.target.matches(".view-lines")){
					let elt = mutation.target.parentElement;
					if(!elt.matches(".lines-content")){
						throw Error("This shouldn't happen.");
					}
					elt.style.contain = "";
					elt.style.overflow = "";
					elt = elt.parentElement;
					if(!elt.matches(".monaco-scrollable-element")){
						throw Error("This shouldn't happen.");
					}
					elt.style.overflow = "";
					this.lineOffsetMutationObserver.observe(
						this.querySelector(".view-lines"),
						{"childList" : true, "attributeFilter" : ["style"], "subtree" : true}
					);
					this.lineOffsetMutationObserver.observe(
						this.querySelector(".view-overlays"),
						{"childList" : true, "attributeFilter" : ["style"], "subtree" : true}
					);
					this.updateLineOffsets();
				}
			}
		});
	}

	connectedCallback(){
        let styles = document.createElement("style");
        styles.innerText = ReplElement.styles;
		this.appendChild(styles);
		this.dummyTextArea = document.createElement("textarea");
		this.dummyTextArea.setAttribute("readonly", "");
		this.dummyTextArea.style.position = "absolute";
		this.dummyTextArea.style.opacity = 0;
		this.appendChild(this.dummyTextArea);
		let div = document.createElement("div");
		this.overflowMutationObserver.observe(div, {"childList" : true, "subtree" : true});
        div.className = "root";
        this.appendChild(div);
		this.editor = monaco.editor.create(
            this.querySelector(".root"),
            this.editorOptions
		);
		this.editor.updateOptions({
			lineNumbers : this._getEditorLinePrefix.bind(this)
		});
		sleep(10).then(() => {
			// Takes a minute for the Dom to update so we need to sleep first.
			this.querySelector(".decorationsOverviewRuler").remove();
		});
		this._resizeObserver = new ResizeObserver(entries => this.editor.layout());
		this._resizeObserver.observe(this);			
		this.editor.onKeyDown(this._onkey.bind(this));
		this.editor.onMouseDown(this._onmousedown.bind(this));
		this.editor.onMouseUp(this._onmouseup.bind(this));
		this.editor.onDidChangeCursorSelection(this._onDidChangeCursorSelection.bind(this));
		this.editor.onDidScrollChange(() => this.updateLineOffsets());
		window.addEventListener("cut", this._oncut.bind(this));
	}

	_getEditorLinePrefix(n){
		if(n in this.firstLines){
			return ">>>";
		}
		if(n in this.outputModelLines){
			return "";
		}
		return "...";
	}

	fixCursorOutputPosition(){
		let cursorLineNumber = this.editor.getPosition().lineNumber;
		this.querySelector(".cursor").style.marginLeft = cursorLineNumber in this.outputModelLines ? "-4ch" : "";
	}

	_onmousedown(){
		this.mouseDown = true;
		this.justSteppedHistory = false;
		this.fixCursorOutputPosition();
	}

	_onmouseup(){
		this.mouseDown = false;
		if(this.doesSelectionIntersectReadOnlyRegion() && this.editor.getSelection().isEmpty()){
			this.editor.setPosition(this.endOfInputPosition);
			this.revealSelection();
			this.fixCursorOutputPosition();
		}
	}	

	async _onDidChangeCursorSelection(e){
		if(e.secondarySelections.length > 0){
			this.editor.setSelection(e.selection);
			if(e.source !== "keyboard"){
				return;
			}
			if(e.reason === monaco.editor.CursorChangeReason.Redo){
				this.undoStep();
				return;
			}
			if(e.reason === monaco.editor.CursorChangeReason.Undo){
				this.redoStep();
				return;
			}
			throw Error("Unreachable?");
		}
		await sleep(0); // Need to sleep(0) to allow this.mouseDown to update.
		if(!this.mouseDown && e.selection.isEmpty() && this.doesCurrentSelectionIntersectReadOnlyRegion()){
			this.editor.setPosition(this.endOfInputPosition);
			this.revealSelection();
		}
		this.fixCursorOutputPosition();
	}


	/** 
	*   e.preventDefault() doesn't work on certain keys, including Backspace, Tab, and Delete.
	* 	As a recourse to prevent these rogue key events, we move the focus out of the input area 
	*  	for just a moment and then move it back.
	*   The .dummy should be a readonly textarea element.
	*/
	async preventKeyEvent(){
		this.dummyTextArea.focus();
		await sleep(0);
		this.focus();
	}


	enforceReadOnlyRegion(e){
		if(e.browserEvent.type !== "keydown"){
			return;
		}
		if(
			(e.browserEvent.ctrlKey || e.browserEvent.altKey || !/^[ -~]$/.test(e.browserEvent.key)) 
			&& !["Backspace", "Tab", "Delete", "Enter"].includes(e.browserEvent.key)
		){
			return;
		}
		this.intersectSelectionWithInputRegion();
	}

	intersectSelectionWithInputRegion(){
		const sel = this.editor.getSelection();
		let newSel = sel.intersectRanges(this.allOfInputSelection);
		if(newSel){
			newSel = this.setSelectionDirection(newSel, sel.getDirection());
			this.editor.setSelection(newSel);
		} else {
			this.editor.setPosition(this.endOfInputPosition);
		}
		this.revealSelection();
		return !newSel || !sel.equalsSelection(newSel);
	}


	_onkey(e){
		if(this.maybeStepHistory(event)){
			this.preventKeyEvent();
			return
		}
		this.justSteppedHistory = false;
		// Many Ctrl+X commands require special handling.
		if(e.browserEvent.ctrlKey){
			if(e.browserEvent.key in ReplElement._ctrlCmdHandlers){
				ReplElement._ctrlCmdHandlers[e.browserEvent.key].call(this, e);
				return
			}
		}
		if(e.browserEvent.key === "PageUp"){
			this._onPageUp(e);
			return;
		}

		// Prevent keys that would edit value if read only
		if(this.readOnly && !e.browserEvent.altKey){
			if(
				!e.browserEvent.ctrlKey && /^[ -~]$/.test(e.browserEvent.key) 
				|| ["Backspace", "Tab", "Delete", "Enter"].includes(e.browserEvent.key)
			){
				this.preventKeyEvent();
				return;
			}
		}
		// Prevent backing up past start of repl input
		if(this.atStartOfInputRegion() && ["ArrowLeft", "Backspace"].includes(e.browserEvent.key)){
			this.preventKeyEvent();
			return;
		}
		this.enforceReadOnlyRegion(e);
		if(e.browserEvent.key === "Enter") {
			if(this.shouldEnterSubmit(e.browserEvent)){
				this.preventKeyEvent();
				this.submit();
			}
		}		
	}

	static get _ctrlCmdHandlers(){
		return {
			"c" : ReplElement.prototype._onCtrlC,
			"v" : ReplElement.prototype._onCtrlV,
			"x" : ReplElement.prototype._onCtrlX,
			"a" : ReplElement.prototype._onCtrlA,
			"Home" : ReplElement.prototype._onCtrlHome
		};
	}

	_onCtrlC() {}

	_onCtrlV(e) {
		if(e.browserEvent.altKey){
			// Ctrl+alt+V does nothing by default.
			return;
		}
		if(this.readOnly){
			this.preventKeyEvent();
		}
		if(this.doesCurrentSelectionIntersectReadOnlyRegion()){
			this.preventKeyEvent();
			this.editor.setPosition(this.endOfInputPosition);
			this.revealSelection();
		}
	}

	_onCtrlX() {
		// In these cases, we do special handling in the "window.oncut" event listener see _oncut
		if(
			this.readOnly ||
			this.doesCurrentSelectionIntersectReadOnlyRegion() 
			|| this.value.indexOf("\n") === -1 && this.editor.getSelection().isEmpty()
		){
			this.preventKeyEvent();
		}
		// Otherwise allow default behavior
		return
	}

	_oncut(e) {
		e.preventDefault();
		if(this.editor.getSelection().isEmpty()){
			if(this.value.indexOf("\n") === -1 || this.readOnly){
				event.clipboardData.setData('text/plain', this.value + "\n");
				this.editor.pushUndoStop();
				this.editor.executeEdits(
					"cut", 
					[{
						range : this.allOfInputSelection,
						text : ""
					}],
					[this.endOfInputSelection]
				);
				this.editor.pushUndoStop();
				this.preventKeyEvent();
				return;
			}
			// Else allow default handling
			return;
		}
		if(this.doesCurrentSelectionIntersectReadOnlyRegion() || this.readOnly){ 
			event.clipboardData.setData('text/plain', this.editor.getModel().getValueInRange(this.editor.getSelection()));
			this.preventKeyEvent();
			return;
		}
		// Else allow default handling
	}

	_onCtrlA(){
		if(this.altKey){
			// By default if alt key is pressed nothing happens.
			return;
		}
		if(this.doesCurrentSelectionIntersectReadOnlyRegion()){
			// If the current selection intersects read only region, allow default behavior
			// which selects everything
			return;
		}
		// Otherwise, we only want to select the input region
		this.editor.setSelection(this.allOfInputSelection);
		this.preventKeyEvent();
	}

	_onCtrlHome(e){
		if(e.altKey){
			// By default nothing happens if alt key is pressed.
			return;
		}
		// If we are in the read only region
		if(this.doesCurrentSelectionIntersectReadOnlyRegion()){
			// And the user is holding shift, allow default behavior
			if(e.browserEvent.shiftKey){
				return;
			}
			// If in read only region but user is not holding shift, move to start of input region
			this.editor.setPosition(this.startOfInputPosition);
			this.revealSelection();
			this.preventKeyEvent();
			return;
		}
		// If in input region, select from start of current selection to start of input region
		if(e.browserEvent.shiftKey){
			this.editor.setSelection(this.editor.getSelection().setEndPosition(this.startOfInputPosition));
		} else {
			this.editor.setPosition(this.startOfInputPosition);
		}
		this.revealSelection();
		this.preventKeyEvent();
	}

	_onPageUp(e){
		if(e.browserEvent.altKey){
			// alt key means move screen not cursor. So default handling is always fine.
			return;
		}
		// If we are in the read only region
		if(this.doesCurrentSelectionIntersectReadOnlyRegion()){
			// And the user is holding shift, allow default behavior
			if(e.browserEvent.shiftKey){
				return;
			}
			// If in read only region but user is not holding shift, move to start of input region
			this.editor.setPosition(this.startOfInputPosition);
			this.revealSelection();
			this.preventKeyEvent();
			return;				
		}
		// If we would page up into read only region, select input region up to beginning.
		let targetLine = this.getScreenLineOfPosition(this.editor.getPosition()) - this.linesPerScreen + 1;
		if(targetLine <= this.getLastScreenLineOfModelLine(this.readOnlyLines)){
			if(e.browserEvent.shiftKey){
				this.editor.setSelection(this.editor.getSelection().setEndPosition(this.startOfInputPosition));
			} else {
				this.editor.setPosition(this.startOfInputPosition);
			}
			this.revealSelection();
			this.preventKeyEvent();
		}
		// Otherwise, allow default page up behavior
	}






	updateLineOffsets(){
		const outputLines = 
			Array.from(document.querySelectorAll(".view-line"))
				.map(e => [e, Number(e.style.top.slice(0, -2))]).sort((a,b) => a[1]-b[1]).map(e => e[0]);
		const topScreenLine = Math.floor(this.editor.getScrollTop() / this.lineHeight) + 1;
		outputLines.forEach((e, idx) => {
			let targetMargin = (topScreenLine + idx) in this.outputScreenLines ? "-4ch" : "";
			if(e.style.marginLeft !== targetMargin){
				e.style.marginLeft = targetMargin;
			}
		});
		const outputOverlays = 
			Array.from(document.querySelector(".view-overlays").children)
				.map(e => [e, Number(e.style.top.slice(0, -2))]).sort((a,b) => a[1]-b[1]).map(e => e[0]);
		outputOverlays.forEach((e, idx) => {
			let targetMargin = (topScreenLine + idx) in this.outputScreenLines ? "-5ch" : "";
			if(e.style.marginLeft !== targetMargin){
				e.style.marginLeft = targetMargin;
			}
		});
	}
	
	async nextHistory(n = 1){
        await this.stepHistory(n);
    }
    
    async previousHistory(n = 1){
        await this.stepHistory(-n);
    }

    async stepHistory(didx) {
		this.history.setTemporaryValue(this.value);
		const changed = this.history.step(didx);
		if(!changed){
			return
		}
		this.editor.pushUndoStop();
		// Use a multi cursor before and after the edit operation to indicate that this is a history item
		// change. In onDidChangeCursorSelection will check for a secondary selection and use the presence 
		// of it to deduce that a history step occurred. We then immediately clear the secondary selection
		// to prevent the user from ever seeing it.
		this.editor.getModel().pushEditOperations(
			[this.editor.getSelection(), new monaco.Selection(1, 1, 1, 1)], 
			[{
				range : this.allOfInputSelection,
				text : (await this.history.value) || ""
			}],
			() => [this.endOfInputSelection, new monaco.Selection(1, 1, 1, 1)]
		);
		this.editor.pushUndoStop();
		// For some reason we end up selecting the input. This is maybe a glitch with pushEditOperations?
		// It doesn't happen with this.editor.executeEdits, but using executeEdits doesn't work because
		// for some reason we can't convince it that the before cursor state is a multi cursor like we need 
		// for this strategy.
		this.editor.setPosition(this.endOfInputPosition);
		this.revealSelection();
		// Record how much the history index changed for undo.
		this.historyIndexRedoStack = [];
		this.historyIndexUndoStack.push(didx);
		this.justSteppedHistory = true;
	}

	
	maybeStepHistory(event){
		let result = this.shouldStepHistory(event);
		if(result){
			let dir = {"ArrowUp" : -1, "ArrowDown" : 1 }[event.key];
			this.stepHistory(dir);
		}
		return result;
	}

    shouldStepHistory(event){
		if(!["ArrowUp", "ArrowDown"].includes(event.key)){
			return false;
		}
		if(this.justSteppedHistory){
			return true;
		}
		if(!this.editor.getSelection().isEmpty()){
			return false;
		}
		let pos = this.editor.getPosition();
        if(event.key === "ArrowUp"){
			return this.getScreenLineOfPosition(pos) === this.getScreenLineOfPosition(this.startOfInputPosition);
        }
        if(event.key === "ArrowDown"){
            return this.getScreenLineOfPosition(pos) === this.getScreenLineOfPosition(this.endOfInputPosition);
        }
        throw Error("Unreachable");
	}

	shouldEnterSubmit(event){
		if(event.ctrlKey){
			return true;
		}
		if(event.shiftKey){
			return false;
		}
		if(this.value.search("\n") >= 0){
			return false;
		}
		if(/(:|\\)\s*$/.test(this.value)){
			return false;
		}
        return true;
	}


	async submit(){
		const code = this.value;
		if(!code.trim()){
			return;
		}
		this.readOnly = true;
		let syntaxCheck = await this.executor.validate(code);
		if(!syntaxCheck.validated){
			this.showSyntaxError(syntaxCheck.error);
			this.readOnly = false;
			return;
		}
        this.history.push(code);
        await sleep(0);
		this.historyIdx = this.history.length;
		const data = await this.executor.execute(code);
		const totalLines = this.editor.getModel().getLineCount();
		let outputLines = data.result_repr !== undefined ? 1 : 0;

		this.readOnlyLines = totalLines + outputLines;
		this.firstLines[this.readOnlyLines + 1] = true;
		
        if(data.result_repr !== undefined){
			this.addOutput(data.result_repr);
		} else {
			this.editor.setValue(`${this.editor.getValue()}\n`);
		}
		this.editor.setPosition(this.endOfInputPosition);
		this.revealSelection();
		this.readOnly = false;
	}

    addOutput(value){
		const totalScreenLines = this.getScreenLineCount();
		const totalModelLines = this.getModelLineCount();
		this.editor.setValue(`${this.editor.getValue()}\n${value}\n`);
		this.outputModelLines[totalModelLines + 1] = true;
		let numScreenLines = this.getScreenLinesInModelLine(totalModelLines + 1);
		for(let curLine = totalScreenLines + 1; curLine <= totalScreenLines + numScreenLines; curLine++){
			this.outputScreenLines[curLine] = true;
		}
	}
	
	async showSyntaxError(error){
		let line = this.readOnlyLines + error.lineno;
		let col = error.offset;
		let decorations = [{
			range: new monaco.Range(line, 1, line, 1),
			options: {
				isWholeLine: true,
				afterContentClassName: 'repl-error repl-error-decoration',
			}
		}];
		if(col){
			let endCol = this.editor.getModel().getLineMaxColumn(line);
			decorations.push({
				range: new monaco.Range(line, col, line, endCol),
				options: {
					isWholeLine: true,
					className: 'repl-error repl-error-decoration-highlight',
					inlineClassName : 'repl-error-decoration-text'

				}
			});
		}
		this.decorations = this.editor.deltaDecorations([], decorations);
		this.errorWidget = {
			domNode: null,
			getId: function() {
				return 'error.widget';
			},
			getDomNode: function() {
				if (!this.domNode) {
					this.domNode = document.createElement('div');
					this.domNode.innerText = `${error.type}: ${error.msg}`;
					this.domNode.classList.add("repl-error");
					this.domNode.classList.add("repl-error-widget");
					this.domNode.setAttribute("transition", "show");
					promiseFromDomEvent(this.domNode, "transitionend").then(
						() => this.domNode.removeAttribute("transition")
					);
					sleep(0).then(() => this.domNode.setAttribute("active", ""));
				}
				return this.domNode;
			},
			getPosition: function() {
				return {
					position: {
						lineNumber: line,
						column: 1
					},
					preference: [monaco.editor.ContentWidgetPositionPreference.BELOW]
				};
			}
		};
		this.editor.addContentWidget(this.errorWidget);
		await sleep(10);
		let elts = [
			document.querySelector(".repl-error-decoration-highlight"),
			document.querySelector(".repl-error-decoration"),
		];
		for(let elt of elts){
			elt.setAttribute("transition", "show");
			elt.setAttribute("active", "");
			promiseFromDomEvent(elt, "transitionend").then(
				() => elt.removeAttribute("transition")
			);
		}
	}
}
customElements.define('repl-terminal', ReplElement);